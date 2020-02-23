import os
import argparse
import json
import time
import joblib
import datetime
from pathlib import Path

import numpy as np
from itertools import islice


def chunks(data, traj_len = 20, slide = 5):
    #it = iter(data)
    for i in range(0, len(data), slide):
        yield {k:data[k] for k in islice(data, i, i + traj_len)}


def parsing_into_chunks(data, args, seq_name):
    
    # process each chunk of traj_length number of frames (traj_len = 20 etc)
    out_data = []
    for chunk in chunks(data, traj_len = args.traj_len, slide = args.slide):
        chunk_dict = {}

        # skip chunk if it is not continous
        listFrameIds = list(map(int,  list(chunk)))
        if(listFrameIds[0] + args.traj_len-1 != listFrameIds[-1]):
            print("a chunk with start frame {} is not continous".format(listFrameIds[0]))
            continue

        # find common person ids in chunk
        allPIds = sum([chunkData["pIds"] for chunkData in chunk.values()],[])
        allPIds = list(map(int, allPIds))
        allPoses = sum([chunkData["pose"] for chunkData in chunk.values()],[])
        allScales = sum([chunkData["scale"] for chunkData in chunk.values()],[])
        allGtLocations = sum([chunkData["gt_locations"] for chunkData in chunk.values()],[])
        allOpenPoseLocations = sum([chunkData["mid_hip"] for chunkData in chunk.values()],[])
        allGridFlow = [chunkData["gridFlow"] for chunkData in chunk.values()]

        # Extract data for each person
        pIds, cPose, cScale, cOpenPoseLocation, cGridFlow, cGtLocation  = [], [], [], [], [], []
        for pid in np.unique(allPIds): 

            pidIndexes =  np.where(allPIds == pid)[0]
            if(len(pidIndexes) < args.traj_len):
                continue    # skip if trajectory of this person is short
            
            pPose = [allPoses[i] for i in pidIndexes] 
            pScale = [allScales[i] for i in pidIndexes] 
            pOpenPoseLocation = [allOpenPoseLocations[i] for i in pidIndexes] 
            pGridFlow = allGridFlow
            pGtLocation = [allGtLocations[i] for i in pidIndexes] 

            pIds.append(pid)
            cPose.append(pPose)
            cScale.append(pScale)
            cOpenPoseLocation.append(pOpenPoseLocation)
            cGridFlow.append(pGridFlow)
            cGtLocation.append(pGtLocation)


        if(len(pIds) == 0): continue
        chunk_dict["start_frame"] =  listFrameIds[0]
        chunk_dict["seq_name"] =  seq_name
        chunk_dict["pIds"] = np.asarray(pIds)
        chunk_dict["pose"] = np.asarray(cPose)     # ~[num_ped, traj_len, 50]
        chunk_dict["openPoseLocation"] = np.asarray(cOpenPoseLocation)   # ~[num_ped, traj_len, 2] 
        chunk_dict["scale"] = np.expand_dims(np.asarray(cScale),axis = 2)  # ~[num_ped, traj_len, 1] 
        chunk_dict["gt_location"] = np.asarray(cGtLocation)   # ~[num_ped, traj_len, 2] 
        chunk_dict["gridFlow"] = np.asarray(cGridFlow)   # ~[num_ped, traj_len, 24] 

        out_data.append(chunk_dict)

    return  out_data  

def gather_data(raw_data_dir, seq_name, args):

    with open(os.path.join(raw_data_dir,"{}_openpose_location.json".format(seq_name)), "r") as f:
        openPoseLocation = json.load(f)
    with open(os.path.join(raw_data_dir,"{}_location_gt.json".format(seq_name)), "r") as f:
        gtLocation = json.load(f)
    with open(os.path.join(raw_data_dir,"{}_openpose_pose_18.json".format(seq_name)), "r") as f:
        pose = json.load(f)
    with open(os.path.join(raw_data_dir,"{}_openpose_scale.json".format(seq_name)), "r") as f:
        scale = json.load(f)
    with open(os.path.join(raw_data_dir,"{}_gridflow.json".format(seq_name)), "r") as f:
        gridFlow = json.load(f)

    seqDict = {}
    seqDict["start_frame"], seqDict["seq_name"], seqDict["pIds"], seqDict["pose"], \
    seqDict["scales"], seqDict["egomotions"], seqDict["gt_location"], \
    seqDict["openPoseLocation"] = [], [], [], [], [], [], [], []
    
    # get list of frame numbers 
    listFrameNumbers = list(openPoseLocation.keys())
    listFrameNumbers = list(map(int, listFrameNumbers)) 

    for fIndex, fId in enumerate(listFrameNumbers[:-args.traj_len:args.slide]): 
        if str(fId) not in gridFlow:
            # skip if this frame does not have optical flow data
            continue

        if(listFrameNumbers[fIndex] + args.traj_len -1 != listFrameNumbers[fIndex+ args.traj_len -1] ):
            #skip if frames are not continuous
            print("skip frame", fId)
            continue

        # find all pids in frames from fId to fId+19
        listPids = [openPoseLocation[str(f)]["pIds"] for f in range(fId, fId + args.traj_len)]
        listPids = np.unique(np.array(sum(listPids,[])))
        # gather data for each pedestrian in listPids
        for pId in listPids:
            skip = False
            # get features for each trajectory of a pedestrian
            pose_sample, scale_sample, ego_sample, gtlocations_sample, \
            openPoseLocation_sample = [], [], [], [], []
            for f in range(fId,fId+20):
                f = str(f)
                if(pId not in openPoseLocation[f]["pIds"]):
                    # skip if pid is not in the frame
                    skip = True
                    continue

                pindex = openPoseLocation[f]["pIds"].index(pId)
                pose_sample.append(pose[f]["pose"][pindex]) 
                scale_sample.append(scale[f]["scale"][pindex])
                ego_sample.append(gridFlow[f])
                gtPIdIndx = gtLocation[f]["pIds"].index(pId)
                #gtlocations_sample.append(openPoseLocation[f]["mid_hip"][pindex])
                gtlocations_sample.append(gtLocation[f]["locations"][gtPIdIndx])
                openPoseLocation_sample.append(openPoseLocation[f]["mid_hip"][pindex]) 

            if(not skip):
                seqDict["start_frame"].append(fId) 
                seqDict["seq_name"].append(seq_name) 
                seqDict["pIds"].append(pId) 
                seqDict["pose"].append(pose_sample)
                seqDict["scales"].append(scale_sample)
                seqDict["egomotions"].append(ego_sample)
                seqDict["gt_location"].append(gtlocations_sample)
                seqDict["openPoseLocation"].append(openPoseLocation_sample)

    return seqDict

if __name__ == "__main__":

    # Read arguments. 
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, help ="directory containing raw data")
    parser.add_argument('--output_dir', type=str, help ="directory containing processed data")
    parser.add_argument('--data_file', type=str, help ="file containing training or testing sequences")
    parser.add_argument('--mode', type=str, help ="train/test mode")
    parser.add_argument('--traj_len', type=int, default= 20, help ="trajectory length ")
    parser.add_argument('--slide', type=int, default=1, help =" sliding size")
    parser.add_argument('--ratio', type=float, default=0.8, help ="train/validation ratio ")
    args = parser.parse_args()
    if(args.mode == "test"): args.ratio = 1  # use all data for testing - no split.

    # Read list of sequence names
    with open(args.data_file, "r") as f:
        listseq_names = f.read()
        listseq_names = listseq_names.splitlines()

    # Process each video sequence data and split them into 2 splits.
    dataSplit1, dataSplit2 = {}, {}

    for seq_name in listseq_names:
        seqData = gather_data(args.raw_data_dir, seq_name, args)          # gather feature data.
        #seqChunks = parsing_into_chunks(seqData, args, seq_name)    # parse data into chunks (e.g 20 frames).
        num_samples = len(seqData["start_frame"])
        for key, data in seqData.items():
            if(key not in dataSplit1):
                dataSplit1[key] = []
                dataSplit2[key] = []

            dataSplit1[key].append(seqData[key][:round(args.ratio*num_samples)])   # split data using ratio
            dataSplit2[key].append(seqData[key][round(args.ratio*num_samples):])


        print("seq_name={}   #num_samples={}   #split1/split2:{}/{}"
            .format(seq_name, num_samples , args.ratio*num_samples, num_samples - args.ratio*num_samples))

    # merge data in each key 
    for key in dataSplit1:
        dataSplit1[key] = np.array(sum(dataSplit1[key],[]))
        dataSplit2[key] = np.array(sum(dataSplit2[key],[]))

    dataSplit1["scales"] = np.expand_dims( dataSplit1["scales"],axis=2) 
    ''' Shape of features
    start_frame, seq_name, pIds ~ [num_samples,]
    pose ~ [num_sample, traj_len, 50]
    scales ~[num_sample, traj_len, 1]
    egomotions ~[num_sample, traj_len, 24]
    gt_location ~[num_sample, traj_len, 2]
    openPoseLocation ~ [num_sample, traj_len, 2]
    '''

    '''Write to files '''
    if(args.mode == "train"):

        print("Total samples: train /evaluation={}/{}".format(len(dataSplit1["seq_name"]), len(dataSplit2["seq_name"])))
        trainFilename = os.path.join(args.output_dir, "train_data.joblib")
        valFilename = os.path.join(args.output_dir, "val_data.joblib")

        if not os.path.exists(os.path.dirname(trainFilename)):
            os.makedirs(os.path.dirname(trainFilename))          

        joblib.dump(dataSplit1, trainFilename)
        print("Written train data to {}".format(trainFilename))

        joblib.dump(dataSplit2, valFilename)
        print("Written validation data to {}".format(valFilename))

    else:

        print("Total samples: test={}".format(dataSplit1["seq_name"].shape[0]))
        testFilename = os.path.join(args.output_dir, "test_data.joblib")
        if not os.path.exists(os.path.dirname(testFilename)):
            os.makedirs(os.path.dirname(testFilename))     
        joblib.dump(dataSplit1, testFilename)
        print("Written test data to {}".format(testFilename))


    print("Completed sucessfully")