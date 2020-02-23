import time
import numpy as np
import json 
import torch
from torch import nn, optim
from torch.autograd import Variable

from utils.logger import Logger
from utils.get_dataset_loader import get_dataset_loader
from utils.evaluate import calculate_error, evaluate

from config import get_args

from adf_utils import adf_get_model, adf_test_multiple_network, \
                      adf_save_model, adf_load_model, adf_train_kSamples


'''Read input arguments'''
args = get_args()                                  
logger = Logger(args, mode="test")                 # log 
logger.write("{}\n".format(args))
np.random.seed(seed=1701)

'''Create train/val datasets'''
val_dataset, val_loader = get_dataset_loader(args, args.test_data, flip=False, shuffle=False)

if __name__ == '__main__':

   # create a list of models 
    listModels = []                          
 
    # create train model
    parentInfo = adf_get_model(args, time_stamp=0, model_id=0)    # load model 
    listModels.append(parentInfo)                                 # add parent model to the list.

    # init 
    vLoss, vADE, vFDE = 0, 0, 0
    total_test_time, total_train_time, train_time, test_time = 0, 0, 0, 0
    kSamples = []
    res_list = []

    # start testing
    for itr, sample in enumerate(val_loader):
       
        traj_in = Variable(sample["traj_in"].type(torch.FloatTensor))             # [sample_size, obs_len , 2]
        traj_gt = Variable(sample["traj_gt"].type(torch.FloatTensor))             # [sample_size, pred_len, 2]
        if(args.use_cuda): 
            traj_in, traj_gt =  traj_in.cuda(), traj_gt.cuda()

        # test all models
        test_start = time.time()
        loss, traj_pred, best_model_id = adf_test_multiple_network(listModels, traj_in, traj_gt)
        test_time = (time.time() - test_start)*1000

        # validate the best model 
        ADE, FDE, traj_pred_abs = evaluate(traj_in, traj_gt, traj_pred, val_dataset)
        vLoss += loss
        vADE += ADE.item()
        vFDE  += FDE.item()

        # save train model if this model performs best. 
        if(args.max_models > 1 and best_model_id == 0):
            listModels = adf_save_model(listModels, listModels[0], itr, args)

        # Train using k past samples 
        if(itr + 1 == 1): start_train = True      # which iteration you want to start continuous train ?
        if(args.num_past_chunks > 0 and start_train):

            if(best_model_id != 0):
                listModels = adf_load_model(listModels, listModels[best_model_id], itr, args)   

            # collect recent num_past_chunks 
            kSamples.append(sample)          # list of list
            kSamples = kSamples[1:] if(len(kSamples) > args.num_past_chunks) else kSamples # remove the first one in list.
            
            # train ksamples
            train_start = time.time() 
            train_loss, listModels[0] = adf_train_kSamples(listModels[0], kSamples, args)
            train_time = (time.time() - train_start)*1000


        total_test_time += test_time
        total_train_time += train_time

        logger.write("sample={}/{}   ".format(itr + 1,len(val_dataset)) +\
             "start_frame={}   ".format(sample["start_frame"]) +\
             "seq_name={}   ".format(sample["seq_name"]) +\
             "sample_ADE={:.2f}    ".format(ADE) +\
             "sample_FDE={:.2f}    ".format(FDE) +\
             "sample_loss={:.5f}    ".format(loss) +\
             "avg_ADE ={:.2f}    ".format(vADE/(itr + 1 )) +\
             "avg_FDE={:.2f}    ".format(vFDE/(itr + 1 )) +\
             "avg_loss={:.5f}    ".format(vLoss/(itr + 1 )) +\
             "train_loss={:.5f}   ".format(train_loss if args.num_past_chunks and start_train else -1) +\
             "best_model_id={}   ".format(best_model_id) +\
             "num_models={}   ".format(len(listModels)) + \
             "#train_samples={}   ".format(len(kSamples)) +\
             "test_time(ms)={:.2f}   ".format(test_time) +\
             "train_time(ms)={:.2f}   ".format(train_time) +\
             "avg_test_time(ms)={:.2f}   ".format(total_test_time/(itr + 1)) +\
             "avg_train_time(ms)={:.2f}   ".format(total_train_time/(itr + 1))
            )


        # save trajectories
        res = {} 
        res["seq_name"] = sample["seq_name"]
        res["pId"] = sample["pId"].data.tolist()
        res["start_frame"] = sample["start_frame"].data.tolist()
        res["traj_in_abs"] = sample["traj_in_abs"].data.tolist()
        res["traj_gt_abs"] = sample["traj_gt_abs"].data.tolist()
        res["pred_traj"] = traj_pred_abs.tolist()
        res_list.append(res)

    if(args.traj_file != ""):
        with open(args.traj_file, "w") as write_file:
            json.dump(res_list, write_file)
