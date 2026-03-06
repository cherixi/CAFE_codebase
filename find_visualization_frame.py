import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple


FrameKey = Tuple[int, int, int]  # (vid, sid, fid)


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    gid: int
    act: int
    score: float


@dataclass
class Group:
    act: int
    members: Set[int]
    score: float


def read_labelmap_ids(labelmap_path: Optional[str]) -> List[int]:
    if not labelmap_path:
        return []
    class_ids: List[int] = []
    with open(labelmap_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("  id:") or line.startswith("  label_id:"):
                class_ids.append(int(line.strip().split(" ")[-1]))
    return sorted(set(class_ids))


def parse_txt(path: str, is_pred: bool) -> Dict[FrameKey, List[Det]]:
    frames: Dict[FrameKey, List[Det]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            row = line.strip().split()
            if not row:
                continue
            if len(row) < 9:
                raise ValueError(f"{path}:{ln} 至少需要 9 列，当前 {len(row)} 列")
            v_id, s_id, f_id = int(row[0]), int(row[1]), int(row[2])
            x1, y1, x2, y2 = map(float, row[3:7])
            gid = int(row[7])
            act = int(row[8])
            score = float(row[9]) if is_pred and len(row) >= 10 else 1.0
            frames[(v_id, s_id, f_id)].append(Det(x1, y1, x2, y2, gid, act, score))
    return frames


def group_iou(a: Set[int], b: Set[int]) -> float:
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a) + len(b) - inter)


def make_groups(frame_dets: Sequence[Det]) -> List[Group]:
    members_by_gid: Dict[int, List[int]] = defaultdict(list)
    act_by_gid: Dict[int, int] = {}
    score_by_gid: Dict[int, List[float]] = defaultdict(list)
    for idx, det in enumerate(frame_dets):
        members_by_gid[det.gid].append(idx)
        if det.gid not in act_by_gid:
            act_by_gid[det.gid] = det.act
        score_by_gid[det.gid].append(det.score)

    groups: List[Group] = []
    for gid, members in members_by_gid.items():
        groups.append(
            Group(
                act=act_by_gid[gid],
                members=set(members),
                score=sum(score_by_gid[gid]) / max(len(score_by_gid[gid]), 1),
            )
        )
    return groups


def class_match_counts(
    pred_groups: Sequence[Group], gt_groups: Sequence[Group], class_id: int, thresh: float
) -> Tuple[int, int, int]:
    p = [g for g in pred_groups if g.act == class_id and len(g.members) >= 2]
    g = [g for g in gt_groups if g.act == class_id and len(g.members) >= 2]
    p = sorted(p, key=lambda x: x.score, reverse=True)

    matched = [False] * len(g)
    tp = 0
    fp = 0
    for pg in p:
        best_i = -1
        best_iou = 0.0
        for i, gg in enumerate(g):
            if matched[i]:
                continue
            iou = group_iou(pg.members, gg.members)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0 and best_iou >= thresh:
            matched[best_i] = True
            tp += 1
        else:
            fp += 1
    fn = len(g) - tp
    return tp, fp, fn


def outlier_set(groups: Sequence[Group], outlier_id: int) -> Set[int]:
    ids: Set[int] = set()
    for g in groups:
        if g.act == outlier_id:
            ids |= g.members
    return ids


def frame_score(
    pred_frame: Sequence[Det],
    gt_frame: Sequence[Det],
    class_ids: Sequence[int],
    outlier_id: int,
    w_1_0: float,
    w_0_5: float,
    w_outlier: float,
) -> Tuple[float, Dict[str, float]]:
    pred_groups = make_groups(pred_frame)
    gt_groups = make_groups(gt_frame)

    tp10 = fp10 = fn10 = 0
    tp05 = fp05 = fn05 = 0
    for cid in class_ids:
        tp, fp, fn = class_match_counts(pred_groups, gt_groups, cid, thresh=1.0)
        tp10 += tp
        fp10 += fp
        fn10 += fn
        tp, fp, fn = class_match_counts(pred_groups, gt_groups, cid, thresh=0.5)
        tp05 += tp
        fp05 += fp
        fn05 += fn

    denom10 = tp10 + fp10 + fn10
    denom05 = tp05 + fp05 + fn05
    score10 = (tp10 / denom10) if denom10 > 0 else 1.0
    score05 = (tp05 / denom05) if denom05 > 0 else 1.0

    p_out = outlier_set(pred_groups, outlier_id)
    g_out = outlier_set(gt_groups, outlier_id)
    out_union = len(p_out | g_out)
    out_iou = (len(p_out & g_out) / out_union) if out_union > 0 else 1.0

    total = w_1_0 * score10 + w_0_5 * score05 + w_outlier * out_iou
    detail = {
        "score_1.0": score10,
        "score_0.5": score05,
        "outlier_iou": out_iou,
        "tp_1.0": float(tp10),
        "fp_1.0": float(fp10),
        "fn_1.0": float(fn10),
        "tp_0.5": float(tp05),
        "fp_0.5": float(fp05),
        "fn_0.5": float(fn05),
    }
    return total, detail


def gt_complexity(
    gt_frame: Sequence[Det],
    outlier_id: int,
) -> Dict[str, float]:
    groups = make_groups(gt_frame)
    non_outlier_groups = [g for g in groups if g.act != outlier_id and len(g.members) >= 2]
    non_outlier_actors = {i for i, d in enumerate(gt_frame) if d.act != outlier_id}
    group_sizes = sorted([len(g.members) for g in non_outlier_groups], reverse=True)
    largest_group_ratio = (
        (group_sizes[0] / max(len(non_outlier_actors), 1)) if group_sizes else 1.0
    )
    return {
        "gt_non_outlier_groups": float(len(non_outlier_groups)),
        "gt_non_outlier_actors": float(len(non_outlier_actors)),
        "gt_largest_group_ratio": largest_group_ratio,
    }


def grouping_error_profile(
    pred_frame: Sequence[Det],
    gt_frame: Sequence[Det],
    outlier_id: int,
) -> Dict[str, float]:
    n = min(len(pred_frame), len(gt_frame))
    gt_non_out = {i for i in range(n) if gt_frame[i].act != outlier_id}
    pred_non_out = {i for i in range(n) if pred_frame[i].act != outlier_id}
    stable_non_out = gt_non_out & pred_non_out
    outlier_flip = len(gt_non_out ^ pred_non_out)

    pair_total = 0
    pair_mismatch = 0
    misassigned_actor_ids: Set[int] = set()
    for i, j in combinations(sorted(stable_non_out), 2):
        pair_total += 1
        gt_same = gt_frame[i].gid == gt_frame[j].gid
        pred_same = pred_frame[i].gid == pred_frame[j].gid
        if gt_same != pred_same:
            pair_mismatch += 1
            misassigned_actor_ids.add(i)
            misassigned_actor_ids.add(j)

    mismatch_rate = (pair_mismatch / pair_total) if pair_total > 0 else 0.0
    return {
        "outlier_flip_count": float(outlier_flip),
        "stable_non_out_actors": float(len(stable_non_out)),
        "pair_total": float(pair_total),
        "pair_mismatch_count": float(pair_mismatch),
        "pair_mismatch_rate": mismatch_rate,
        "misassigned_actor_count": float(len(misassigned_actor_ids)),
    }


def select_candidates(
    gt_frames: Dict[FrameKey, List[Det]],
    model_frames: Dict[str, Dict[FrameKey, List[Det]]],
    class_ids: Sequence[int],
    outlier_id: int,
    topk: int,
    strict_middle: bool,
    w_1_0: float,
    w_0_5: float,
    w_outlier: float,
    min_gt_non_outlier_groups: int,
    min_gt_non_outlier_actors: int,
    max_gt_largest_group_ratio: float,
    min_pair_total: int,
    max_outlier_flip_baseline: int,
    max_outlier_flip_full: int,
    min_pair_mismatch_baseline: int,
    require_full_pair_mismatch_lower: bool,
) -> List[Dict]:
    common = set(gt_frames.keys())
    for frames in model_frames.values():
        common &= set(frames.keys())

    rows: List[Dict] = []
    for key in sorted(common):
        gt = gt_frames[key]
        complexity = gt_complexity(gt, outlier_id)
        if complexity["gt_non_outlier_groups"] < min_gt_non_outlier_groups:
            continue
        if complexity["gt_non_outlier_actors"] < min_gt_non_outlier_actors:
            continue
        if complexity["gt_largest_group_ratio"] > max_gt_largest_group_ratio:
            continue

        score_map: Dict[str, float] = {}
        detail_map: Dict[str, Dict[str, float]] = {}
        profile_map: Dict[str, Dict[str, float]] = {}
        for name, frames in model_frames.items():
            s, d = frame_score(frames[key], gt, class_ids, outlier_id, w_1_0, w_0_5, w_outlier)
            score_map[name] = s
            detail_map[name] = d
            profile_map[name] = grouping_error_profile(frames[key], gt, outlier_id)

        b = score_map["baseline"]
        h = score_map["hoi"]
        m = score_map["mae"]
        f = score_map["full"]
        if strict_middle:
            ok_order = (b < h < f) and (b < m < f)
        else:
            ok_order = (b <= h <= f) and (b <= m <= f) and (f > b)
        if not ok_order:
            continue

        bprof = profile_map["baseline"]
        fprof = profile_map["full"]
        if bprof["pair_total"] < min_pair_total:
            continue
        if bprof["outlier_flip_count"] > max_outlier_flip_baseline:
            continue
        if fprof["outlier_flip_count"] > max_outlier_flip_full:
            continue
        if bprof["pair_mismatch_count"] < min_pair_mismatch_baseline:
            continue
        if require_full_pair_mismatch_lower and fprof["pair_mismatch_count"] >= bprof["pair_mismatch_count"]:
            continue

        gap = f - b
        middle_margin = min(h, m) - b
        rows.append(
            {
                "key": key,
                "score_baseline": b,
                "score_hoi": h,
                "score_mae": m,
                "score_full": f,
                "gap_full_baseline": gap,
                "middle_margin": middle_margin,
                "details": detail_map,
                "profiles": profile_map,
                "complexity": complexity,
            }
        )

    rows.sort(
        key=lambda r: (
            r["gap_full_baseline"],
            r["profiles"]["baseline"]["pair_mismatch_count"] - r["profiles"]["full"]["pair_mismatch_count"],
            r["middle_margin"],
            r["score_full"],
        ),
        reverse=True,
    )
    return rows[:topk]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从四个预测txt和gt中筛选更适合可视化的复杂帧"
    )
    parser.add_argument("--gt", required=True, help="gt txt 路径")
    parser.add_argument("--baseline", required=True, help="baseline 预测 txt 路径")
    parser.add_argument("--hoi", required=True, help="仅 HOI graph 预测 txt 路径")
    parser.add_argument("--mae", required=True, help="仅 MAE 预测 txt 路径")
    parser.add_argument("--full", required=True, help="全模型预测 txt 路径")
    parser.add_argument(
        "--labelmap",
        default="./label_map/group_action_list.pbtxt",
        help="label map 路径",
    )
    parser.add_argument("--topk", type=int, default=20, help="输出候选前K帧")
    parser.add_argument(
        "--strict_middle",
        action="store_true",
        help="严格要求 baseline < hoi/mae < full",
    )
    parser.add_argument("--w_1_0", type=float, default=0.4, help="group@1.0 权重")
    parser.add_argument("--w_0_5", type=float, default=0.4, help="group@0.5 权重")
    parser.add_argument("--w_outlier", type=float, default=0.2, help="outlier IoU 权重")

    parser.add_argument("--min_gt_non_outlier_groups", type=int, default=2, help="GT 至少多少个非outlier组(每组>=2人)")
    parser.add_argument("--min_gt_non_outlier_actors", type=int, default=6, help="GT 至少多少个非outlier actor")
    parser.add_argument("--max_gt_largest_group_ratio", type=float, default=0.7, help="GT 最大组人数占非outlier人数比例上限")

    parser.add_argument("--min_pair_total", type=int, default=6, help="用于组错分判断的最少actor对数")
    parser.add_argument("--max_outlier_flip_baseline", type=int, default=0, help="baseline允许的outlier互换人数上限")
    parser.add_argument("--max_outlier_flip_full", type=int, default=0, help="full允许的outlier互换人数上限")
    parser.add_argument("--min_pair_mismatch_baseline", type=int, default=1, help="baseline最少组错分pair数")
    parser.add_argument("--no_require_full_pair_mismatch_lower", action="store_true", help="不强制full的组错分pair少于baseline")

    parser.add_argument("--save_csv", default="", help="可选，保存候选结果csv路径")
    args = parser.parse_args()

    weight_sum = args.w_1_0 + args.w_0_5 + args.w_outlier
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError("权重之和必须为 1.0")

    gt_frames = parse_txt(args.gt, is_pred=False)
    model_frames = {
        "baseline": parse_txt(args.baseline, is_pred=True),
        "hoi": parse_txt(args.hoi, is_pred=True),
        "mae": parse_txt(args.mae, is_pred=True),
        "full": parse_txt(args.full, is_pred=True),
    }

    class_ids = read_labelmap_ids(args.labelmap)
    if not class_ids:
        all_gt_acts = [det.act for dets in gt_frames.values() for det in dets]
        max_act = max(all_gt_acts)
        class_ids = list(range(1, max_act))
    outlier_id = max(class_ids) + 1

    candidates = select_candidates(
        gt_frames=gt_frames,
        model_frames=model_frames,
        class_ids=class_ids,
        outlier_id=outlier_id,
        topk=args.topk,
        strict_middle=args.strict_middle,
        w_1_0=args.w_1_0,
        w_0_5=args.w_0_5,
        w_outlier=args.w_outlier,
        min_gt_non_outlier_groups=args.min_gt_non_outlier_groups,
        min_gt_non_outlier_actors=args.min_gt_non_outlier_actors,
        max_gt_largest_group_ratio=args.max_gt_largest_group_ratio,
        min_pair_total=args.min_pair_total,
        max_outlier_flip_baseline=args.max_outlier_flip_baseline,
        max_outlier_flip_full=args.max_outlier_flip_full,
        min_pair_mismatch_baseline=args.min_pair_mismatch_baseline,
        require_full_pair_mismatch_lower=(not args.no_require_full_pair_mismatch_lower),
    )

    if not candidates:
        print("没有找到满足条件的帧。")
        print("建议：适当放宽过滤条件，例如提高 max_outlier_flip_* 或降低 min_gt_non_outlier_*。")
        return

    best = candidates[0]
    v, s, f = best["key"]
    bprof = best["profiles"]["baseline"]
    fprof = best["profiles"]["full"]
    cpx = best["complexity"]

    print("最推荐可视化帧:")
    print(f"  vid={v}, sid={s}, fid={f}")
    print(
        "  scores: "
        f"baseline={best['score_baseline']:.4f}, "
        f"hoi={best['score_hoi']:.4f}, "
        f"mae={best['score_mae']:.4f}, "
        f"full={best['score_full']:.4f}"
    )
    print(f"  full-baseline gap={best['gap_full_baseline']:.4f}")
    print(
        "  complexity: "
        f"groups={int(cpx['gt_non_outlier_groups'])}, "
        f"non_out_actors={int(cpx['gt_non_outlier_actors'])}, "
        f"largest_group_ratio={cpx['gt_largest_group_ratio']:.3f}"
    )
    print(
        "  grouping_error (baseline -> full): "
        f"pair_mismatch={int(bprof['pair_mismatch_count'])} -> {int(fprof['pair_mismatch_count'])}, "
        f"outlier_flip={int(bprof['outlier_flip_count'])} -> {int(fprof['outlier_flip_count'])}"
    )

    print("\nTop candidates:")
    for i, row in enumerate(candidates, start=1):
        vv, ss, ff = row["key"]
        bp = row["profiles"]["baseline"]
        fp = row["profiles"]["full"]
        cx = row["complexity"]
        print(
            f"{i:02d}. ({vv},{ss},{ff}) "
            f"B={row['score_baseline']:.4f} H={row['score_hoi']:.4f} M={row['score_mae']:.4f} F={row['score_full']:.4f} "
            f"gap={row['gap_full_baseline']:.4f} "
            f"GTg={int(cx['gt_non_outlier_groups'])} GTn={int(cx['gt_non_outlier_actors'])} "
            f"pair_err={int(bp['pair_mismatch_count'])}->{int(fp['pair_mismatch_count'])} "
            f"out_flip={int(bp['outlier_flip_count'])}->{int(fp['outlier_flip_count'])}"
        )

    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(
                [
                    "vid",
                    "sid",
                    "fid",
                    "score_baseline",
                    "score_hoi",
                    "score_mae",
                    "score_full",
                    "gap_full_baseline",
                    "middle_margin",
                    "gt_non_outlier_groups",
                    "gt_non_outlier_actors",
                    "gt_largest_group_ratio",
                    "baseline_pair_mismatch_count",
                    "full_pair_mismatch_count",
                    "baseline_outlier_flip_count",
                    "full_outlier_flip_count",
                ]
            )
            for row in candidates:
                vv, ss, ff = row["key"]
                bp = row["profiles"]["baseline"]
                fp = row["profiles"]["full"]
                cx = row["complexity"]
                writer.writerow(
                    [
                        vv,
                        ss,
                        ff,
                        row["score_baseline"],
                        row["score_hoi"],
                        row["score_mae"],
                        row["score_full"],
                        row["gap_full_baseline"],
                        row["middle_margin"],
                        int(cx["gt_non_outlier_groups"]),
                        int(cx["gt_non_outlier_actors"]),
                        cx["gt_largest_group_ratio"],
                        int(bp["pair_mismatch_count"]),
                        int(fp["pair_mismatch_count"]),
                        int(bp["outlier_flip_count"]),
                        int(fp["outlier_flip_count"]),
                    ]
                )
        print(f"\n已保存候选列表到: {args.save_csv}")


if __name__ == "__main__":
    main()
