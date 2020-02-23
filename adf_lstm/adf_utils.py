import random
import concurrent.futures
import numpy as np
import copy

import torch
from torch import nn, optim
from torch.autograd import Variable
from models.LSTM import LSTM
from utils.dataset import std_denormalize, rel_to_nor
from torchsummary import summary



def adf_get_model(args, time_stamp, model_id):

    # Create model 
    model = LSTM(obs_len=args.obs_len, pred_len=args.pred_len, use_cuda=args.use_cuda)
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.learning_rate)

    if(args.use_cuda) : model = model.cuda() 
    if(args.resume_model != ""):
        model.load_state_dict(torch.load(args.resume_model))

    return {'model': model, 'optimizer': optimizer, 'time_stamp': time_stamp, \
            'model_id': model_id}

def adf_test(modelInfo, traj_in, traj_gt):

    model = modelInfo['model']
    model.eval()

    traj_pred, loss = model(traj_in, traj_gt)                        # forward         
   
    return traj_pred, loss
    
def adf_test_multiple_network(listModels, traj_in, traj_gt):

    best_loss = 1e6

    # Test each model in listModels and select the best one.
    for modelInfo in listModels:
        traj_pred, loss = adf_test(modelInfo, traj_in, traj_gt)
    
        if(loss.item() < best_loss):
           best_loss = loss.item()
           best_traj_pred = traj_pred
           best_model_id = modelInfo['model_id']

    return best_loss, best_traj_pred, best_model_id


def adf_find_least_recent_used_model(listModels):

    lru_timestamp = 1e6       # randomly big number
    for modelInfo in listModels:
        if(modelInfo["time_stamp"] < lru_timestamp and modelInfo["model_id"] != 1):
            lru_timestamp = modelInfo["time_stamp"]
            lru_modelInfo = modelInfo 

    return lru_modelInfo

def adf_save_model(listModels, parentInfo, time_stamp, args):
    

    #serializers.save_npz("parent_{}_{}_temp.pt".format(args.eval_split,args.num_past_chunks), parentInfo["model"])

    if(len(listModels) == args.max_models):

        # if number of models reaches to the maximum
        # then replace the least recently used (LRU)
        if(args.replacement_policy == 'LRU'):
            lru_modelInfo =  adf_find_least_recent_used_model(listModels)
            lru_modelInfo['time_stamp'] = time_stamp
            lru_modelInfo['model'].load_state_dict(parentInfo["model"].state_dict())
    else: 
        # else, add new modelwork to the end of model list.
            # Save the current modelwork

        childInfo =  adf_get_model(args, time_stamp=time_stamp, model_id= len(listModels))
        childInfo['model'].load_state_dict(parentInfo["model"].state_dict())
        listModels.append(childInfo)

    return listModels

def adf_load_model(listModels, bestModelInfo, itr, args):

    #serializers.save_npz("best_{}_{}_temp.pt".format(args.eval_split, args.num_past_chunks), bestModelInfo["model"])
    #serializers.load_npz("best_{}_{}_temp.pt".format(args.eval_split, args.num_past_chunks), listModels[0]['model'])
    listModels[0]['model'].load_state_dict(bestModelInfo["model"].state_dict())

    return listModels

def adf_train(modelInfo, traj_in, traj_gt):

    model = modelInfo['model']
    optimizer = modelInfo['optimizer']

    model.train()
    optimizer.zero_grad()                                            # zero out gradients
    traj_pred, loss = model(traj_in, traj_gt)                        # forward         
    loss.backward()                                                  # backward       
    #torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step() 
 
    return traj_pred, loss


def divide_chunks(l, n): 
    # split a list into chunks 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def adf_train_kSamples(parentInfo, kSamples, args):
    

    args.num_epoch_cn = int(min(len(kSamples)/20 + 2, 30))
    for _ in range(args.num_epoch_cn):

        # Shuffle kSamples and splits them into batches
        if(len(kSamples) > args.batch_size_cn):
            random.shuffle(kSamples) #shuffle method
        listSamples =  list(divide_chunks(kSamples, args.batch_size_cn))

        for samples in listSamples:

            # gather samples with same keys before feeding into network.
            batch_samples = {}
            for k in samples[0].keys():
                batch_samples[k] = [sample[k] for sample in samples]
                batch_samples[k] = np.vstack(batch_samples[k])    

            traj_in = Variable(torch.FloatTensor(batch_samples["traj_in"]))
            traj_gt = Variable(torch.FloatTensor(batch_samples["traj_gt"]))

            if(args.use_cuda): 
                traj_in, traj_gt =  traj_in.cuda(), traj_gt.cuda()
    

            traj_pred, loss = adf_train(parentInfo, traj_in, traj_gt)

    return loss, parentInfo