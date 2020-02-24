import os
import numpy as np
import json


def save_trajectories(eval_data, eval_result, res_file):
    '''save resulted trajectories into json file  '''

    '''prepare json output file'''

    #print(eval_result["pred_traj"].size())
    res = {} 
    res["seqname_list"] = eval_data["seqname_list"].tolist()
    res["start_list"] = eval_data["start_list"].tolist()
    res["offset_list"] = eval_data["offset_list"].tolist()
    res["traj_list"] = eval_data["traj_list"].tolist()
    res["pred_traj"] = eval_result["pred_traj"].tolist()

    with open(res_file, "a") as write_file:
        json.dump(res, write_file)

    #input("here")