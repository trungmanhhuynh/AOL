import os
import numpy as np
import json


def save_trajectories(sample, pred_traj, traj_file):
    '''save resulted trajectories into json file  '''

    '''prepare json output file'''

    #print(eval_result["pred_traj"].size())
    res = {} 
    res["seq_name"] = sample["seq_name"]
    res["start_frame"] = sample["start_frame"].data.tolist()
    res["traj_in_abs"] = sample["traj_in_abs"].data.tolist()
    res["traj_gt_abs"] = sample["traj_gt_abs"].data.tolist()
    res["pred_traj"] = pred_traj.tolist()

    with open(traj_file, "a") as write_file:
        json.dump(res, write_file)

    #input("here")