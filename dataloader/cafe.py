import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import json
import numpy as np
import random
from PIL import Image

ACTIVITIES = ['Queueing', 'Ordering', 'Eating/Drinking', 'Working/Studying', 'Fighting', 'TakingSelfie']


# read annotation files
def cafe_read_annotations(path, videos, num_class):
    labels = {}
    group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    for vid in videos:
        video_path = os.path.join(path, vid)
        for cid in os.listdir(video_path):
            clip_path = os.path.join(video_path, cid)            
            label_path = clip_path + '/ann.json'

            with open(label_path, 'r') as file:
                groups = {}
                boxes, actions, activities, members, membership = [], [], [], [], []

                values = json.load(file)
                num_frames = values['framesCount']
                frame_interval = values['framesEach']
                actors = values['figures']

                key_frame = actors[0]['shapes'][0]['frame']

                for i, actor in enumerate(actors):
                    actor_idx = actor['id']
                    group_name = actor['label']

                    box = actor['shapes'][0]['coordinates']
                    x1, y1 = box[0]
                    x2, y2 = box[1]
                    x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    boxes.append([x_c, y_c, w, h])

                    if group_name != 'individual':
                        group_idx = int(group_name[-1])
                        if actor['attributes'][0]['value'] != "":
                            action = group_to_id[actor['attributes'][0]['value']['key']]

                            if group_idx not in groups.keys():
                                groups[group_idx] = {'activity': action}

                            if 'members' in groups[group_idx].keys():
                                groups[group_idx]['members'][i] = 1
                            else:
                                groups[group_idx]['members'] = torch.zeros(len(actors))
                                groups[group_idx]['members'][i] = 1
                        else:
                            if group_idx in groups.keys():
                                action = groups[group_idx]['activity']
                            else:
                                action = -1
                    else:
                        action = num_class
                        group_idx = 0

                    actions.append(action)
                    membership.append(group_idx)

                for i, action in enumerate(actions):
                    if action == -1:
                        group_idx = membership[i]

                        if group_idx in groups.keys():
                            new_action = groups[group_idx]['activity']
                            actions[i] = new_action

                            group_members = groups[group_idx]['members']
                            group_members[i] = 1
                        else:
                            membership[i] = 0
                            actions[i] = num_class

                for group_id in sorted(groups):
                    if group_id - 1 >= len(groups):
                        new_id = len(groups)

                        while new_id > 0:
                            if new_id not in groups:
                                groups[new_id] = groups[group_id]
                                del groups[group_id]
                                for i in range(len(membership)):
                                    if membership[i] == group_id:
                                        membership[i] = new_id
                                group_id = new_id
                            new_id -= 1

                for group_id in sorted(groups):
                    activities.append(groups[group_id]['activity'])
                    members.append(groups[group_id]['members'])

                actions = np.array(actions, dtype=np.int32)
                boxes = np.vstack(boxes)
                membership = np.array(membership, dtype=np.int32) - 1
                activities = np.array(activities, dtype=np.int32)

                actions = torch.from_numpy(actions).long()
                boxes = torch.from_numpy(boxes).float()
                membership = torch.from_numpy(membership).long()
                activities = torch.from_numpy(activities).long()

                if len(members) == 0:
                    members = torch.tensor(members)
                else:
                    members = torch.stack(members).float()

                annotations = {
                    'boxes': boxes,
                    'actions': actions,
                    'membership': membership,
                    'activities': activities,
                    'members': members,
                    'num_frames': num_frames,
                    'interval': frame_interval,
                    'key_frame': key_frame,
                }

            if len(annotations['activities']) != 0:
                labels[(int(vid), int(cid))] = annotations

    return labels


def cafe_all_frames(labels):
    frames = []

    for sid, anns in labels.items():
        frames.append(sid)
    return frames


class CafeDataset(data.Dataset):
    def __init__(self, frames, anns, tracks, image_path, args, is_training=True, object_tracks=None):
        super(CafeDataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.tracks = tracks
        self.object_tracks = object_tracks
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.num_boxes = args.num_boxes
        self.use_olic = bool(getattr(args, 'use_olic', False))
        self.use_object_context = self.use_olic or bool(getattr(args, 'use_interaction_stir', False))
        self.num_object_boxes = int(getattr(args, 'num_object_boxes', 10))
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_class = args.num_class
        self.is_training = is_training
        self.videomae_feats_path = getattr(args, 'videomae_feats_path', None)
        self.use_mae = getattr(args, 'use_mae', False)
        self.mae_dim = getattr(args, 'mae_dim', 768)
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_object_frame_rows(self, vid, cid, fid):
        if self.object_tracks is None:
            return np.zeros((0, 10), dtype=np.float32)
        clip_tracks = self.object_tracks.get((vid, cid))
        if clip_tracks is None:
            return np.zeros((0, 10), dtype=np.float32)

        if isinstance(clip_tracks, dict):
            rows = clip_tracks.get(fid, [])
        elif isinstance(clip_tracks, list):
            if fid < 0 or fid >= len(clip_tracks):
                rows = []
            else:
                rows = clip_tracks[fid]
        else:
            rows = []

        if rows is None or len(rows) == 0:
            return np.zeros((0, 10), dtype=np.float32)

        arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return np.zeros((0, 10), dtype=np.float32)
        return arr

    def _parse_object_rows(self, vid, cid, fid):
        m = self.num_object_boxes
        boxes = np.zeros((m, 4), dtype=np.float32)      # normalized xyxy
        valid = np.zeros((m,), dtype=np.float32)
        scores = np.zeros((m,), dtype=np.float32)
        token_ids = np.zeros((m,), dtype=np.int64)
        family_ids = np.zeros((m,), dtype=np.int64)

        rows = self._get_object_frame_rows(vid, cid, fid)
        if rows.shape[0] == 0:
            return boxes, valid, scores, token_ids, family_ids

        take = min(rows.shape[0], m)
        for i in range(take):
            row = rows[i]
            if row.shape[0] < 5:
                continue
            x1 = float(row[1])
            y1 = float(row[2])
            x2 = float(row[3])
            y2 = float(row[4])
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            row_valid = float(row[9]) if row.shape[0] >= 10 else 1.0
            if row_valid <= 0.5:
                continue

            boxes[i] = np.array([x1, y1, x2, y2], dtype=np.float32)
            valid[i] = 1.0
            scores[i] = float(row[5]) if row.shape[0] >= 6 else 0.0
            family_ids[i] = int(row[6]) if row.shape[0] >= 7 else 0
            token_ids[i] = int(row[8]) if row.shape[0] >= 9 else 0

        return boxes, valid, scores, token_ids, family_ids

    def __getitem__(self, idx):
        if self.num_frame == 1:
            frames = self.select_key_frames(self.frames[idx])
        else:
            frames = self.select_frames(self.frames[idx])

        samples = self.load_samples(frames)

        return samples

    def __len__(self):
        return len(self.frames)

    def select_key_frames(self, frame):
        annotation = self.anns[frame]
        key_frame = annotation['key_frame']

        return [(frame, int(key_frame))]

    def select_frames(self, frame):
        annotation = self.anns[frame]
        key_frame = annotation['key_frame']
        total_frames = annotation['num_frames']
        interval = annotation['interval']

        if self.is_training:
            # random sampling
            if self.random_sampling:
                sample_frames = random.sample(range(total_frames), self.num_frame)
                sample_frames.sort()
            # segment-based sampling
            else:                
                segment_duration = total_frames // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)
        else:
            # random sampling
            if self.random_sampling:
                sample_frames = random.sample(range(total_frames), self.num_frame)
                sample_frames.sort()
            # segment-based sampling
            else:
                segment_duration = total_frames // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)

        return [(frame, int(fid * annotation['interval'])) for fid in sample_frames]

    def load_samples(self, frames):
        images, boxes, gt_boxes, actions, activities, members, membership = [], [], [], [], [], [], []
        object_boxes_xyxy, object_valid_mask, object_scores = [], [], []
        object_token_id, object_family_id = [], []
        targets = {}
        fids = []

        for i, (frame, fid) in enumerate(frames):
            vid, cid = frame
            fids.append(fid)
            img = Image.open(self.image_path + '/%s/%s/images/frames_%d.jpg' % (vid, cid, fid))
            image_w, image_h = img.width, img.height
            img = self.transform(img)
            images.append(img)

            num_boxes = self.anns[frame]['boxes'].shape[0]

            for box in self.anns[frame]['boxes']:
                x_c, y_c, w, h = box
                gt_boxes.append([x_c / image_w, y_c / image_h, w / image_w, h / image_h])

            temp_boxes = np.ones((num_boxes, 4))
            for j, track in enumerate(self.tracks[(vid, cid)][fid]):                
                _id, x1, y1, x2, y2 = track

                if x1 < 0.0 and y2 < 0.0:
                    x1, y1, x2, y2 = 0.0, 0.0, 1e-8, 1e-8

                x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1

                if _id <= num_boxes:
                    temp_boxes[int(_id - 1)] = np.array([x_c, y_c, w, h])

            boxes.append(temp_boxes)
            actions = [self.anns[frame]['actions']]
            activities = [self.anns[frame]['activities']]
            members = [self.anns[frame]['members']]
            membership = [self.anns[frame]['membership']]

            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], (self.num_boxes - len(boxes[-1])) * [[0.0, 0.0, 0.0, 0.0]]])

            if len(actions[-1]) != self.num_boxes:
                actions[-1] = torch.cat((actions[-1], torch.tensor((self.num_boxes - len(actions[-1])) * [self.num_class + 1])))

            if members[-1].shape[1] != self.num_boxes:
                members[-1] = torch.hstack(
                    (members[-1], torch.zeros((members[-1].shape[0], self.num_boxes - members[-1].shape[1]))))

            if len(membership) != self.num_boxes:
                membership[-1] = torch.cat((membership[-1], torch.tensor((self.num_boxes - len(membership[-1])) * [-1])))

            if self.use_object_context:
                obj_boxes, obj_valid, obj_score, obj_token, obj_family = self._parse_object_rows(
                    vid=vid, cid=cid, fid=fid
                )
                object_boxes_xyxy.append(obj_boxes)
                object_valid_mask.append(obj_valid)
                object_scores.append(obj_score)
                object_token_id.append(obj_token)
                object_family_id.append(obj_family)

        images = torch.stack(images)
        boxes = np.vstack(boxes).reshape([self.num_frame, -1, 4])
        gt_boxes = np.vstack(gt_boxes).reshape([self.num_frame, -1, 4])
        actions = torch.stack(actions)
        membership = torch.stack(membership)

        if len(activities) == 0:
            activities = torch.tensor(activities)
            members = torch.tensor(activities)
        else:
            activities = torch.stack(activities)
            members = torch.stack(members)

        boxes = torch.from_numpy(boxes).float()
        gt_boxes = torch.from_numpy(gt_boxes).float()

        targets['actions'] = actions
        targets['activities'] = activities
        targets['boxes'] = boxes
        targets['gt_boxes'] = gt_boxes
        targets['members'] = members
        targets['membership'] = membership

        if self.use_object_context:
            targets['object_boxes_xyxy'] = torch.from_numpy(
                np.stack(object_boxes_xyxy, axis=0).astype(np.float32)
            )
            targets['object_valid_mask'] = torch.from_numpy(
                np.stack(object_valid_mask, axis=0).astype(np.float32)
            )
            targets['object_scores'] = torch.from_numpy(
                np.stack(object_scores, axis=0).astype(np.float32)
            )
            targets['object_token_id'] = torch.from_numpy(
                np.stack(object_token_id, axis=0).astype(np.int64)
            )
            targets['object_family_id'] = torch.from_numpy(
                np.stack(object_family_id, axis=0).astype(np.int64)
            )

        if self.use_mae and self.videomae_feats_path:
            feat_path = os.path.join(self.videomae_feats_path, f'{vid}_{cid}.npy')
            if os.path.exists(feat_path):
                mae_feats = np.load(feat_path)
                if mae_feats.ndim == 2:
                    mae_feats = mae_feats.squeeze(0)
                targets['mae_feats'] = torch.from_numpy(mae_feats).float()
            else:
                targets['mae_feats'] = torch.zeros(self.mae_dim).float()

        infos = {'vid': vid, 'sid': cid, 'fid': fids, 'key_frame': self.anns[frame]['key_frame']}

        return images, targets, infos
