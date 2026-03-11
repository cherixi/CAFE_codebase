import argparse
import csv
import json
from pathlib import Path

import numpy as np


def box_stats(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    inlier = values[(values >= lower) & (values <= upper)]
    if inlier.size == 0:
        whisker_low = float(np.min(values))
        whisker_high = float(np.max(values))
    else:
        whisker_low = float(np.min(inlier))
        whisker_high = float(np.max(inlier))
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "max": float(np.max(values)),
        "iqr": float(iqr),
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
    }


def load_groups(path: Path):
    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=False)
        groups = {k: z[k] for k in z.files}
        z.close()
        return groups
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 1:
            return {path.stem: arr}
        if arr.ndim == 2 and arr.shape[1] == 2:
            return {"x": arr[:, 0], "y": arr[:, 1]}
        return {f"{path.stem}_flat": arr.reshape(-1)}
    raise ValueError(f"Unsupported file type: {path}")


def save_outputs(rows, out_prefix: Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "count",
                "min",
                "q1",
                "median",
                "q3",
                "max",
                "iqr",
                "whisker_low",
                "whisker_high",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Export boxplot elements (min/q1/median/q3/max + whiskers) from .npy/.npz files."
    )
    parser.add_argument("--input", required=True, type=str, help="Path to .npy or .npz")
    parser.add_argument(
        "--out_prefix",
        default="",
        type=str,
        help="Output path prefix (without extension). Default: <input_stem>_boxstats next to input.",
    )
    parser.add_argument(
        "--drop_empty",
        action="store_true",
        help="Skip groups with zero valid values",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    groups = load_groups(in_path)
    rows = []
    for name, arr in groups.items():
        stat = box_stats(arr)
        if stat is None:
            if args.drop_empty:
                continue
            stat = {
                "count": 0,
                "min": None,
                "q1": None,
                "median": None,
                "q3": None,
                "max": None,
                "iqr": None,
                "whisker_low": None,
                "whisker_high": None,
            }
        stat["group"] = name
        rows.append(stat)

    out_prefix = Path(args.out_prefix) if args.out_prefix else in_path.with_name(f"{in_path.stem}_boxstats")
    json_path, csv_path = save_outputs(rows, out_prefix)
    print("Saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
