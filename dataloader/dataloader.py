# ------------------------------------------------------------------------
# Modified from SSU (https://github.com/cvlab-epfl/social-scene-understanding)
# Modified from ARG (https://github.com/wjchaoGit/Group-Activity-Recognition)
# ------------------------------------------------------------------------
from .cafe import *

import pickle
import os
import json
import sys


def _load_pickle_compat(pkl_path):
    """
    Load pickle with a small compatibility shim for numpy internal module path
    changes (e.g. numpy._core.numeric vs numpy.core.numeric).
    """
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        msg = str(e)
        if "numpy._core" not in msg:
            raise

        # Compatibility fallback for pickles produced in a different numpy major version.
        import numpy as np

        try:
            sys.modules.setdefault("numpy._core", np.core)
            if hasattr(np.core, "numeric"):
                sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        except Exception:
            raise e

        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

TRAIN_CAFE_P = ['1', '2', '3', '4', '9', '10', '11', '12', '17', '18', '19', '20', '21', '22', '23', '24']
VAL_CAFE_P = ['13', '14', '15', '16']
TEST_CAFE_P = ['5', '6', '7', '8']

TRAIN_CAFE_V = ['1', '2', '5', '6', '9', '10', '13', '14', '17', '18', '21', '22']
VAL_CAFE_V = ['3', '7', '11', '15', '19', '23']
TEST_CAFE_V = ['4', '8', '12', '16', '20', '24']


def _load_tracks_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    tracks = {}
    for clip in payload.get('tracks', []):
        vid = int(clip['vid'])
        cid = int(clip['cid'])
        frames = clip.get('frames', [])
        # Keep the same indexing behavior as pkl: tracks[(vid, cid)][fid]
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


def read_dataset(args):
    if args.dataset == 'cafe':
        data_path = os.path.join(args.data_path, 'cafe')
        print(f"    Data path: {data_path}")

        # split-by-place setting
        if args.split == 'place':
            TRAIN_VIDEOS_CAFE = TRAIN_CAFE_P
            VAL_VIDEOS_CAFE = VAL_CAFE_P
            TEST_VIDEOS_CAFE = TEST_CAFE_P
            print(f"    Using split-by-place")
        # split-by-view setting
        elif args.split == 'view':
            TRAIN_VIDEOS_CAFE = TRAIN_CAFE_V
            VAL_VIDEOS_CAFE = VAL_CAFE_V
            TEST_VIDEOS_CAFE = TEST_CAFE_V
            print(f"    Using split-by-view")
        else:
            assert False

        if args.val_mode:
            print(f"    Val mode: using TRAIN for training, VAL for testing")
            train_data = cafe_read_annotations(data_path, TRAIN_VIDEOS_CAFE, args.num_class)
            train_frames = cafe_all_frames(train_data)

            test_data = cafe_read_annotations(data_path, VAL_VIDEOS_CAFE, args.num_class)
            test_frames = cafe_all_frames(test_data)
        else:
            print(f"    Normal mode: using TRAIN+VAL for training, TEST for testing")
            print(f"    Loading training data from videos: {TRAIN_VIDEOS_CAFE + VAL_VIDEOS_CAFE}")
            train_data = cafe_read_annotations(data_path, TRAIN_VIDEOS_CAFE + VAL_VIDEOS_CAFE, args.num_class)
            train_frames = cafe_all_frames(train_data)
            print(f"    Loaded {len(train_frames)} training clips")

            print(f"    Loading test data from videos: {TEST_VIDEOS_CAFE}")
            test_data = cafe_read_annotations(data_path, TEST_VIDEOS_CAFE, args.num_class)
            test_frames = cafe_all_frames(test_data)
            print(f"    Loaded {len(test_frames)} test clips")

        # actor tracklets for all frames
        tracks_source = getattr(args, 'tracks_source', 'gt')
        tracks_pkl_path_arg = getattr(args, 'tracks_pkl_path', '')

        if tracks_pkl_path_arg:
            tracks_pkl_path = tracks_pkl_path_arg
            print(f"    Track source: custom pkl path")
        else:
            if tracks_source == 'pred':
                tracks_pkl_path = os.path.join(data_path, 'pred_tracks_aligned_to_gt_slots.pkl')
            else:
                tracks_pkl_path = os.path.join(data_path, 'gt_tracks.pkl')
            print(f"    Track source: {tracks_source}")

        tracks_json_path = os.path.join(data_path, 'gt_tracks.json')

        # Priority: pkl first, then json fallback.
        if os.path.exists(tracks_pkl_path):
            print(f"    Loading actor tracklets from: {tracks_pkl_path}")
            all_tracks = _load_pickle_compat(tracks_pkl_path)
            print(f"    Tracklets loaded (pkl)")
        elif os.path.exists(tracks_json_path):
            print(f"    pkl not found, fallback to json: {tracks_json_path}")
            all_tracks = _load_tracks_from_json(tracks_json_path)
            print(f"    Tracklets loaded (json)")
        else:
            raise FileNotFoundError(
                f"Neither track pkl nor json exists. Checked pkl: {tracks_pkl_path}, json: {tracks_json_path}"
            )

        use_olic = bool(getattr(args, 'use_olic', False))
        object_tracks = None
        if use_olic:
            object_tracks_pkl = getattr(args, 'object_tracks_pkl', '')
            if not object_tracks_pkl:
                object_tracks_pkl = os.path.join(data_path, 'object_tracks_gdino_swinb.pkl')
            if not os.path.exists(object_tracks_pkl):
                raise FileNotFoundError(
                    f"use_olic=True but object track pkl not found: {object_tracks_pkl}"
                )
            print(f"    Loading object tracks from: {object_tracks_pkl}")
            object_tracks = _load_pickle_compat(object_tracks_pkl)
            if not isinstance(object_tracks, dict):
                raise TypeError(
                    f"object_tracks_pkl should be dict, got {type(object_tracks)} from {object_tracks_pkl}"
                )

        print(f"    Creating dataset objects...")
        train_set = CafeDataset(
            train_frames, train_data, all_tracks, data_path, args,
            is_training=True, object_tracks=object_tracks,
        )
        test_set = CafeDataset(
            test_frames, test_data, all_tracks, data_path, args,
            is_training=False, object_tracks=object_tracks,
        )
    else:
        assert False

    print("%d train samples and %d test samples" % (len(train_frames), len(test_frames)))

    return train_set, test_set
