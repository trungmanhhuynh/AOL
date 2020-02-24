# Adapted script create_datasets.py from 
# Future Person Localization in First-Person Videos

from __future__ import print_function
from __future__ import division
from six.moves import range

import os
import argparse
import json
import time
import joblib
import datetime
import quaternion
from functools import reduce
from operator import add

import numpy as np


def read_sfmlearn(ego_path, flip):
    """
    Right-hand coordinate (following SfMLearn paper).
    X: Pitch, Y: Yaw, Z: Roll
    """
    ego_dict = {}
    with open(ego_path, "r") as f:
        for line in f:
            strings = line.strip("\r\n").split(",")
            key = strings[0]
            vx, vy, vz, rx, ry, rz = list(map(lambda x:float(x), strings[1:]))
            if flip:
                ego_dict[key] = np.array([rx, -ry, -rz, -vx, vy, vz])
            else:
                ego_dict[key] = np.array([rx, ry, rz, vx, vy, vz])
    return ego_dict


def read_gridflow(ego_path, flip):
    ego_dict = {}
    with open(ego_path) as f:
        for line in f:
            strings = line.strip("\r\n").split(",")
            key = strings[0]
            grid_flow = np.array(list(map(float, strings[1:])))
            if flip:
                grid_flow[range(0, len(grid_flow), 2)] *= -1
            ego_dict[key] = grid_flow
    return ego_dict


def read_vid_list(indir_list):
    vid_list, nb_image_list = [], []
    blacklist = {}
    with open(indir_list) as f:
        for line in f:
            strs = line.strip("\n").split(",")
            if strs[0].startswith("#"):
                continue
            vid_list.append(strs[0])
            nb_image_list.append(int(strs[1]))
            if len(strs) > 2:
                blacklist[strs[0]] = strs[2:]

    return vid_list, nb_image_list, blacklist

def calc_scales(poses):

    # poses.shape ~ [num_samples, traj_len, 18, 2]
    spine = (poses[:, :, 8:9, :2] + poses[:, :, 11:12, :2]) / 2
    neck = poses[:, :, 1:2, :2]
    scales_all = np.linalg.norm(neck - spine, axis=3)  # (N, T, 1)
    scales_all[scales_all < 1e-8] = 1e-8  # Avoid ZerodivisionError
                                          # scales_all.shape ~ [num_samples, traj_len, 1]

    return scales_all

def accumulate_egomotion(rots, vels):
    # Accumulate translation and rotation
    egos = []
    qa = np.quaternion(1, 0, 0, 0)
    va = np.array([0., 0., 0.])
    for rot, vel in zip(rots, vels):
        vel_rot = quaternion.rotate_vectors(qa, vel)
        va += vel_rot
        qa = qa * quaternion.from_rotation_vector(rot)
        egos.append(np.concatenate(
            (quaternion.as_rotation_vector(qa), va), axis=0))
    return egos

def calc_egomotion(video_ids, frames, egomotion_dict, args):

    egomotions = []
    offset = 0
    past_len = args.input_len
    for vid, frame in zip(video_ids, frames):
        ego_dict = egomotion_dict[vid]
        if args.ego_type == "sfm":  # SfMLearner
            rots, vels = [], []
            for frame in range(frame + offset, frame + offset + past_len + args.pred_len):
                key = "rgb_{:05d}.jpg".format(frame)
                key_m1 = "rgb_{:05d}.jpg".format(frame-1)
                rot_vel = ego_dict[key] if key in ego_dict \
                    else ego_dict[key_m1] if key_m1 in ego_dict \
                    else np.array([0., 0., 0., 0., 0., 0.])
                rots.append(rot_vel[:3])
                vels.append(rot_vel[3:6])

            egos = accumulate_egomotion(rots[:past_len], vels[:past_len]) + \
                accumulate_egomotion(rots[past_len:past_len+args.pred_len], vels[past_len:past_len+args.pred_len])
        else:  # Grid optical flow
            raw_egos = [ego_dict["rgb_{:05d}.jpg".format(f)] for f in
                        range(frame + offset, frame + offset + past_len + args.pred_len)]
            egos = [np.sum(raw_egos[:idx+1], axis=0) for idx in range(past_len)] + \
                [np.sum(raw_egos[past_len:past_len+idx+1], axis=0) for idx in range(args.pred_len)]
        egomotions.append(egos)
    egomotions = np.array(egomotions).astype(np.float32)

    return egomotions


if __name__ == "__main__":
    """
    Create dataset for training
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('indir_list', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--eval_split', type=int, default=0)
    parser.add_argument('--traj_length', type=int, default=30)
    parser.add_argument('--input_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--traj_skip', type=int, default=2)
    parser.add_argument('--traj_skip_test', type=int, default=5)
    parser.add_argument('--nb_splits', type=int, default=3)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=960)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--no_flip', action="store_true")
    parser.add_argument('--ego_type', type=str, default="sfm")



    args = parser.parse_args()
    start = time.time()

    vid_list, nb_image_list, blacklist = read_vid_list(args.indir_list)
    print("Number of videos: {}".format(len(vid_list)))

    shuffled_ids = np.copy(vid_list)
    np.random.seed(args.seed)
    np.random.shuffle(shuffled_ids)
    ratio = 1 / args.nb_splits

    nb_videos = len(vid_list)
    split_dict = {}
    for sp, st in enumerate(np.arange(0, 1, ratio)):
        print(shuffled_ids[int(nb_videos*st):int(nb_videos*(st+ratio))])
        for vid in shuffled_ids[int(nb_videos*st):int(nb_videos*(st+ratio))]:
            split_dict[vid] = sp

    video_ids, frames, person_ids, trajectories, poses, splits, \
        turn_mags, trans_mags, pids = [], [], [], [], [], [], [], [], []
    start_dict, traj_dict, pose_dict = {}, {}, {}

    total_frames = 0
    egomotion_dict = {}
    nb_trajs = 0
    nb_traj_list = [0 for _ in range(args.nb_splits)]
    for video_id, nb_images in zip(vid_list, nb_image_list):
        trajectory_path = os.path.join(
            args.data_dir, "trajectories/{}_trajectories_dynamic.json".format(video_id))
        with open(trajectory_path, "r") as f:
            trajectory_dict = json.load(f)

        total_frames += nb_images

        if args.ego_type == "sfm":
            egomotion_path = os.path.join(args.data_dir, "egomotions/{}_egomotion.csv".format(video_id))
            egomotion_dict[video_id] = read_sfmlearn(egomotion_path, False)
        else:
            egomotion_path = os.path.join(args.data_dir, "egomotions/{}_gridflow_24.csv".format(video_id))
            egomotion_dict[video_id] = read_gridflow(egomotion_path, False)

        lr_mag_list = [abs(v[1]) for k, v in sorted(egomotion_dict[video_id].items())]

        start_dict[video_id] = {}
        traj_dict[video_id] = {}
        pose_dict[video_id] = {}

        # pid search
        pids = []
        for pid, info in trajectory_dict.items():
            if video_id in blacklist and pid in blacklist[video_id]:
                print("Blacklist: {} {}".format(video_id, pid))
                continue
            if "traj_sm" not in info:
                continue
            traj = info["traj_sm"]
            pose = info["pose_sm"]
            if len(traj) < args.traj_length:
                continue
            front_cnt = sum([1 if ps[11][0] - ps[8][0] > 0 else 0 for ps in pose])
            pids.append(pid)

        pid_cnt = len(pids)
        nb_trajs += pid_cnt

        traj_cnt = 0
        for pid in pids:
            info = trajectory_dict[pid]
            t_s = info["start"]
            traj = info["traj_sm"]
            pose = info["pose_sm"]
            pid = int(pid)

            if t_s <= 2 or t_s + len(traj) >= nb_images - 1:
                continue

            start_dict[video_id][pid] = info["start"]
            traj_dict[video_id][pid] = info["traj_sm"]
            pose_dict[video_id][pid] = info["pose_sm"]

            def add_sample(split):
                x_max = np.max([x[0] for x in traj[tidx+args.traj_length//2:tidx+args.traj_length]])
                x_min = np.min([x[0] for x in traj[tidx+args.traj_length//2:tidx+args.traj_length]])
                y_max = np.max([x[1] for x in traj[tidx+args.traj_length//2:tidx+args.traj_length]])
                y_min = np.min([x[1] for x in traj[tidx+args.traj_length//2:tidx+args.traj_length]])
                trans_mag = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
                turn_mag = np.max(lr_mag_list[tidx + t_s - 1 + args.traj_length // 2:tidx + t_s - 1 + args.traj_length])

                frames.append(tidx + t_s)
                person_ids.append(pid)
                trajectories.append(traj[tidx:tidx+args.traj_length])
                poses.append(pose[tidx:tidx+args.traj_length])
                splits.append(split)
                video_ids.append(video_id)

                turn_mags.append(turn_mag)
                trans_mags.append(trans_mag)

            # Training set (split 0-4)
            for tidx in range(0, len(traj) - args.traj_length + 1, args.traj_skip):
                add_sample(split_dict[video_id])
                traj_cnt += 1

            # Evaluation set (split 5-9)
            for tidx in range(0, len(traj) - args.traj_length + 1, args.traj_skip_test):
                add_sample(split_dict[video_id] + args.nb_splits)

        print(video_id, nb_images, pid_cnt, traj_cnt)
        nb_traj_list[split_dict[video_id]] += traj_cnt

    splits = np.array(splits)

    #print("Total number of frames: {}".format(total_frames))
    #result_str = ""
    #for sp in range(args.nb_splits):
    #    result_str += "Split {}: {}".format(sp + 1, sum(splits == sp))
    #    if sp < args.nb_splits - 1:
    #       result_str += ", "

    #print(result_str)
    #print("Number of tracklets:", nb_trajs)
    print("Number of samples in each train split", nb_traj_list)

    ''' Below is further processing for fpl dataset to get the correct format
    data for my framework '''

    poses = np.array(poses)             #~ [num_samples, traj_len, 18, 2]
    frames = np.array(frames)           #~ [num_samples]
    person_ids = np.array(person_ids)   #~ [num_samples]
    trajectories = np.array(trajectories) #~ [num_samples, 20 ,2 ]
    splits = np.array(splits)           # indicates the split index of each sample. 
                                        # [0-4] for training,  [0-4] + nb_splits for testing
                                        #~ [num_samples]
    video_ids = np.array(video_ids)         
                               
    # generates scale data 
    scales = calc_scales(np.array(poses))           # scales.shape ~ [num_samples, traj_len, 1]

    # generates egomotion data
    egomotions = calc_egomotion(video_ids, frames, egomotion_dict, args)   # egomotions ~ [num_samples, traj_len, 6]

    # get data for a fold
    train_splits = list(filter(lambda x: x != args.eval_split, range(5)))
    eval_split = args.eval_split + 5

    print(train_splits)
    print(eval_split)

    train_idx = reduce(add, [splits == s for s in train_splits])
    eval_idx = splits == eval_split


    trainFilename = os.path.join(args.output_dir, "train_data_split_{}.joblib".format(args.eval_split))
    valFilename = os.path.join(args.output_dir, "val_data_split_{}.joblib".format(args.eval_split))

    if not os.path.exists(os.path.dirname(trainFilename)):
        os.makedirs(os.path.dirname(trainFilename))          


    print("MYDEBUG: #train/eval samples for split {}= {}/{}".format(args.eval_split, frames[train_idx].shape[0], frames[eval_idx].shape[0]))
    joblib.dump({
        "start_frame": frames[train_idx],
        "seq_name": video_ids[train_idx],
        "pIds": person_ids[train_idx],
        "pose": poses[train_idx],
        "openPoseLocation": trajectories[train_idx],
        "scales": scales[train_idx], 
        "gt_location": trajectories[train_idx],
        "egomotions": egomotions[train_idx]
        }, trainFilename)
    print("Written train data to {}".format(trainFilename))

    joblib.dump({
        "start_frame": frames[eval_idx],
        "seq_name": video_ids[eval_idx],
        "pIds": person_ids[eval_idx],
        "pose": poses[eval_idx],
        "openPoseLocation": trajectories[eval_idx],
        "scales": scales[eval_idx], 
        "gt_location": trajectories[eval_idx],
        "egomotions": egomotions[eval_idx]
        }, valFilename)
    print("Written validation data to {}".format(valFilename))


    print("Completed. Elapsed time: {} (s)".format(time.time()-start))






