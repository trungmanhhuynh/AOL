import os
import argparse
import json
import time
import joblib
import datetime
from pathlib import Path

import numpy as np
from itertools import islice


'''
Descriptions: 

  + Parsing data (locations, poses) of people in sequences into chunks (batches)

Usage: 
python utils/generate_datasets.py --info_file utils/dataset_info.json
'''

def chunks(data, traj_len = 20, slide = 1):
    #it = iter(data)
    for i in range(0, len(data), slide):
        yield {k:data[k] for k in islice(data, i, i + traj_len)}


def parsing_into_chunks(data, args, seqName):
    
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
        chunk_dict["seq_name"] =  seqName
        chunk_dict["pIds"] = np.asarray(pIds)
        chunk_dict["pose"] = np.asarray(cPose)     # ~[num_ped, traj_len, 50]
        chunk_dict["openPoseLocation"] = np.asarray(cOpenPoseLocation)   # ~[num_ped, traj_len, 2] 
        chunk_dict["scale"] = np.expand_dims(np.asarray(cScale),axis = 2)  # ~[num_ped, traj_len, 1] 
        chunk_dict["gt_location"] = np.asarray(cGtLocation)   # ~[num_ped, traj_len, 2] 
        chunk_dict["gridFlow"] = np.asarray(cGridFlow)   # ~[num_ped, traj_len, 24] 

        out_data.append(chunk_dict)

    return  out_data  


def cal_location_mean_var(data_all):

    # gather location data from all samples.
    location_data = [chunk["locations"] for chunk in data_all]
    location_data = np.vstack(location_data)        # (4616, 20, 2)
    location_data = np.concatenate(location_data, axis = 0) # (4616*20, 2)

    mean = location_data.mean(axis = 0)
    var = location_data.std(axis = 0)

    return mean, var

def gather_data(raw_data_dir, data_file):

    with open(os.path.join(raw_data_dir,"{}_openpose_location.json".format(data_file)), "r") as f:
        openPoseLocation = json.load(f)
    with open(os.path.join(raw_data_dir,"{}_location_gt.json".format(data_file)), "r") as f:
        gtLocation = json.load(f)
    with open(os.path.join(raw_data_dir,"{}_openpose_pose.json".format(data_file)), "r") as f:
        pose = json.load(f)
    with open(os.path.join(raw_data_dir,"{}_openpose_scale.json".format(data_file)), "r") as f:
        scale = json.load(f)
    with open(os.path.join(raw_data_dir,"{}_gridflow.json".format(data_file)), "r") as f:
        gridFlow = json.load(f)

    dataDict = {}
    for fId in openPoseLocation: 

        dataDict[fId] = {}
        dataDict[fId]["pIds"] = openPoseLocation[fId]["pIds"]
        dataDict[fId]["mid_hip"] = openPoseLocation[fId]["mid_hip"]
        dataDict[fId]["pose"] = pose[fId]["pose"]
        dataDict[fId]["scale"] = scale[fId]["scale"]
      
        # add corresponding peds' gt location
        gtPIdIndexes = [gtLocation[fId]["pIds"].index(pId) for pId in openPoseLocation[fId]["pIds"]]
        dataDict[fId]["gt_locations"] = [gtLocation[fId]["locations"][i] for i in gtPIdIndexes]

        if fId in gridFlow:
            dataDict[fId]["gridFlow"] = gridFlow[fId]
        else: 
            del dataDict[fId]

    return dataDict

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
        listSeqNames = f.read()
        listSeqNames = listSeqNames.splitlines()

    # Process each video sequence data and split them into 2 splits.
    dataSplit1, dataSplit2 = [], []
    for seqName in listSeqNames:
        
        seqData = gather_data(args.raw_data_dir, seqName)          # gather feature data.
        seqChunks = parsing_into_chunks(seqData, args, seqName)    # parse data into chunks (e.g 20 frames).

        dataSplit1.append(seqChunks[:round(args.ratio*len(seqChunks))])   # split data using ratio
        dataSplit2.append(seqChunks[round(args.ratio*len(seqChunks)):])
        
        print("train_fn={}   #num_chunks={}   #split1/split2:{}/{}"
            .format(seqName, len(seqChunks) , len(dataSplit1[-1]), len(dataSplit2[-1])))

    dataSplit1 = sum(dataSplit1,[])
    dataSplit2 = sum(dataSplit2,[])


    '''Write to files '''
    if(args.mode == "train"):

        print("Total samples: train /evaluation={}/{}".format(len(dataSplit1), len(dataSplit2)))

        trainFilename = os.path.join(args.output_dir, "train_data.joblib")
        valFilename = os.path.join(args.output_dir, "val_data.joblib")

        if not os.path.exists(os.path.dirname(trainFilename)):
            os.makedirs(os.path.dirname(trainFilename))          

        joblib.dump(dataSplit1, trainFilename)
        print("Written train data to {}".format(trainFilename))

        joblib.dump(dataSplit2, valFilename)
        print("Written validation data to {}".format(valFilename))

    else:

        print("Total samples: test={}".format(len(dataSplit1)))
        testFilename = os.path.join(args.output_dir, "test_data.joblib")

        if not os.path.exists(os.path.dirname(testFilename)):
            os.makedirs(os.path.dirname(testFilename))     

        joblib.dump(dataSplit1, testFilename)
        print("Written test data to {}".format(testFilename))


    print("Completed sucessfully")