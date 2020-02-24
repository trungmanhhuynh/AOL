import time
import numpy as np
import cupy as cp
import json
import chainer 
from chainer import optimizers, Variable, cuda, serializers

from utils.logger import Logger
from utils.get_dataset_loader import get_dataset_loader
from utils.dataset import std_denormalize, rel_to_nor
from utils.evaluate import calculate_error, evaluate

from config import get_args
from core import create_model


from adf_utils import adf_get_model, adf_test_multiple_network, \
                      adf_save_model, adf_load_model, adf_train_kSamples
                      

'''Read input arguments'''
args = get_args()                                  
logger = Logger(args, mode="test")                 # log 
logger.write("{}\n".format(args))
np.random.seed(seed=1701)

'''Create train/val datasets'''
val_dataset, val_loader = get_dataset_loader(args, args.val_data, flip=False, shuffle=False)

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
    for itr, samples in enumerate(val_loader):      
       
        # samples is a list of samples. Each sample is a dictionary of feature arrays. 
        batch_samples = {}
        for k in samples[0].keys():
            batch_samples[k] = [sample[k] for sample in samples]
            batch_samples[k] = np.vstack(np.expand_dims(batch_samples[k], axis=0))  

        # prepare features
        traj_in = Variable(batch_samples["traj_in"].astype(np.float32)) # [batch_size, obs_len , 2]
        scale_in = Variable(batch_samples["scale_in"].astype(np.float32)) # [batch_size, obs_len , 2]
        pose_in = Variable(batch_samples["pose_in"].astype(np.float32)) # [batch_size, obs_len , 2]
        ego_in = Variable(batch_samples["ego_in"].astype(np.float32)) # [batch_size, obs_len , 2]
        traj_gt = Variable(batch_samples["traj_gt"].astype(np.float32))         
        scale_gt = Variable(batch_samples["scale_gt"].astype(np.float32))         

        if(args.gpu >=0):
            traj_in.to_gpu(args.gpu); scale_in.to_gpu(args.gpu) ; pose_in.to_gpu(args.gpu)
            ego_in.to_gpu(args.gpu); traj_gt.to_gpu(args.gpu) ; scale_gt.to_gpu(args.gpu)

        # test all models
        test_start = time.time()
        loss, traj_pred, best_model_id = adf_test_multiple_network(listModels, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt)
        test_time = (time.time() - test_start)*1000

        #Calculate ade/fde
        ADE, FDE, traj_pred_abs = evaluate(cuda.to_cpu(traj_in.data), cuda.to_cpu(traj_gt.data), cuda.to_cpu(traj_pred.data), val_dataset)
        vLoss += loss
        vADE += ADE
        vFDE += FDE


        # save train model if this model performs best. 
        if(args.max_models > 1 and best_model_id == 0):
            listModels = adf_save_model(listModels, listModels[0], itr, args)

        # Train using k past samples 
        if(itr + 1 == 1): start_train = True      # which iteration you want to start continuous train ?
        if(args.num_past_chunks > 0 and start_train):

            if(best_model_id != 0):
                listModels = adf_load_model(listModels, listModels[best_model_id], itr, args)   

            # collect recent num_past_chunks 
            kSamples.append(samples[0])          # list of list
            kSamples = kSamples[1:] if(len(kSamples) > args.num_past_chunks) else kSamples # remove the first one in list.
            
            # train ksamples
            train_start = time.time() 
            train_loss, listModels[0] = adf_train_kSamples(listModels[0], kSamples, args)
            train_time = (time.time() - train_start)*1000

        total_test_time += test_time
        total_train_time += train_time

        logger.write("sample={}/{}   ".format(itr + 1,len(val_dataset)) +\
             "start_frame={}   ".format(batch_samples["start_frame"]) +\
             "seq_name={}   ".format(batch_samples["seq_name"]) +\
             "sample_ADE={:.2f}    ".format(ADE) +\
             "sample_FDE={:.2f}    ".format(FDE) +\
             "sample_loss={:.5f}    ".format(loss) +\
             "avg_ADE ={:.2f}    ".format(vADE/(itr + 1 )) +\
             "avg_FDE={:.2f}    ".format(vFDE/(itr + 1 )) +\
             "avg_loss={:.5f}    ".format(vLoss/(itr + 1 )) +\
             "train_loss={:.5f}   ".format(train_loss if args.num_past_chunks > 0 and start_train else -1) +\
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
        res["seq_name"] = batch_samples["seq_name"].tolist()
        res["pId"] = batch_samples["pId"].tolist()
        res["start_frame"] = batch_samples["start_frame"].tolist()
        res["traj_in_abs"] = batch_samples["traj_in_abs"].tolist()
        res["traj_gt_abs"] = batch_samples["traj_gt_abs"].tolist()
        res["pred_traj"] =  traj_pred_abs.tolist()
        res_list.append(res)

    if(args.traj_file != ""):
        with open(args.traj_file, "w") as write_file:
            json.dump(res_list, write_file)