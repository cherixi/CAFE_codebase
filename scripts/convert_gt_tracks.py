import argparse
import json
import os
import pickle


def convert_tracks_pkl_to_json(input_path, output_path):
    with open(input_path, 'rb') as f:
        tracks = pickle.load(f)

    converted = []
    for key in sorted(tracks.keys()):
        vid, cid = key
        frames = tracks[key]
        frame_items = []
        if isinstance(frames, dict):
            frame_iter = sorted((int(fid), fid_tracks) for fid, fid_tracks in frames.items())
        else:
            frame_iter = [(fid, fid_tracks) for fid, fid_tracks in enumerate(frames)]

        for fid, fid_tracks in frame_iter:
            # Ensure plain JSON numeric lists: [[id, x1, y1, x2, y2], ...]
            normalized_tracks = []
            for t in fid_tracks:
                if len(t) >= 5:
                    normalized_tracks.append([
                        int(t[0]),
                        float(t[1]),
                        float(t[2]),
                        float(t[3]),
                        float(t[4]),
                    ])
                else:
                    normalized_tracks.append([float(x) for x in t])

            frame_items.append({
                'fid': int(fid),
                'tracks': normalized_tracks,
            })

        converted.append({
            'vid': int(vid),
            'cid': int(cid),
            'frames': frame_items,
        })

    payload = {
        'version': 1,
        'source': os.path.basename(input_path),
        'tracks': converted,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'))

    return len(converted)


def load_tracks_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    tracks = {}
    for clip in payload.get('tracks', []):
        vid = int(clip['vid'])
        cid = int(clip['cid'])
        frames = clip.get('frames', [])
        if len(frames) == 0:
            frame_tracks = []
        else:
            max_fid = max(int(frame.get('fid', idx)) for idx, frame in enumerate(frames))
            frame_tracks = [[] for _ in range(max_fid + 1)]
            for idx, frame in enumerate(frames):
                fid = int(frame.get('fid', idx))
                frame_tracks[fid] = frame.get('tracks', [])
        tracks[(vid, cid)] = frame_tracks

    return tracks


def _normalize_track(track):
    if len(track) >= 5:
        return [int(track[0]), float(track[1]), float(track[2]), float(track[3]), float(track[4])]
    return [float(x) for x in track]


def verify_alignment(input_path, output_path):
    with open(input_path, 'rb') as f:
        src = pickle.load(f)
    dst = load_tracks_from_json(output_path)

    if set(src.keys()) != set(dst.keys()):
        raise ValueError('Clip keys mismatch between pkl and json')

    for key in src.keys():
        src_frames = src[key]
        if isinstance(src_frames, dict):
            src_map = {int(fid): src_frames[fid] for fid in src_frames.keys()}
            max_fid = max(src_map.keys()) if len(src_map) > 0 else -1
            src_list = [[] for _ in range(max_fid + 1)]
            for fid, fid_tracks in src_map.items():
                src_list[fid] = fid_tracks
        else:
            src_list = list(src_frames)

        dst_list = dst[key]
        if len(src_list) != len(dst_list):
            raise ValueError(f'Frame count mismatch at clip {key}: {len(src_list)} vs {len(dst_list)}')

        for fid in range(len(src_list)):
            src_tracks = [_normalize_track(t) for t in src_list[fid]]
            dst_tracks = [_normalize_track(t) for t in dst_list[fid]]
            if src_tracks != dst_tracks:
                raise ValueError(f'Track mismatch at clip {key}, frame {fid}')


def main():
    parser = argparse.ArgumentParser(description='Convert CAFE gt_tracks.pkl to gt_tracks.json')
    parser.add_argument('--input', required=True, help='Path to gt_tracks.pkl')
    parser.add_argument('--output', default=None, help='Output path, default: sibling gt_tracks.json')
    parser.add_argument('--no_verify', action='store_true', help='Skip post-conversion alignment check')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or os.path.join(os.path.dirname(input_path), 'gt_tracks.json')

    count = convert_tracks_pkl_to_json(input_path, output_path)

    if not args.no_verify:
        verify_alignment(input_path, output_path)
        print('Verification: PASS')

    print(f'Converted {count} clips')
    print(f'Input: {input_path}')
    print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
