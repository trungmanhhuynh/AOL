import random
import concurrent.futures
import numpy as np
import copy
import sys
import chainer 
from chainer import optimizers, Variable, cuda, serializers
from models.cnn import CNN, CNN_ego_scale, CNN_scale, CNN_pose_scale, CNN_ego_pose_scale, CNN_ego

def adf_get_model(args, time_stamp, model_id):

    if(args.model == "cnn"): 
        model = CNN(args.channel_list, args.deconv_list, args.ksize_list, \
                        args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
    elif(args.model == "cnn_scale"):
        model = CNN_scale(args.channel_list, args.deconv_list, args.ksize_list, \
                        args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
    elif(args.model == "cnn_ego"):
        model = CNN_ego(args.channel_list, args.deconv_list, args.ksize_list, \
                        args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
    elif(args.model == "cnn_ego_scale"):
        model = CNN_ego_scale(args.channel_list, args.deconv_list, args.ksize_list, \
                        args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
    elif(args.model == "cnn_pose_scale"):
        model = CNN_pose_scale(args.channel_list, args.deconv_list, args.ksize_list, \
                        args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
    elif(args.model == "cnn_ego_pose_scale"):
        model = CNN_ego_pose_scale(args.channel_list, args.deconv_list, args.ksize_list, \
                        args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
    else: 
        sys.exit("wrong model type")

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    if args.resume_model != "":
        serializers.load_npz(args.resume_model, model)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    
    return {'model': model, 'optimizer': optimizer, 'time_stamp': time_stamp, \
            'model_id': model_id}

def adf_test(modelInfo, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt):

    model = modelInfo['model']
    chainer.config.train = False
    chainer.config.enable_backprop = False
    traj_pred, loss = model(traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt)                        # forward         
   
    return traj_pred, loss
    
def adf_test_multiple_network(listModels, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt):

    best_loss = 1e6

    # Test each model in listModels and select the best one.
    for modelInfo in listModels:
        traj_pred, loss = adf_test(modelInfo, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt)
        loss_cpu = cuda.to_cpu(loss.data)

        if(loss_cpu < best_loss):
           best_loss = loss_cpu
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
            lru_modelInfo['model'] = copy.deepcopy(parentInfo["model"])
    else: 
        # else, add new modelwork to the end of model list.
            # Save the current modelwork

        childInfo =  adf_get_model(args, time_stamp=time_stamp, model_id= len(listModels))
        childInfo['model'] = copy.deepcopy(parentInfo["model"])
        listModels.append(childInfo)

    return listModels

def adf_load_model(listModels, bestModelInfo, itr, args):

    listModels[0]['model']= copy.deepcopy(bestModelInfo["model"])

    return listModels

def adf_train(modelInfo, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt):

    model = modelInfo['model']
    optimizer = modelInfo['optimizer']

    chainer.config.train = True
    chainer.config.enable_backprop = True
    model.cleargrads()
    traj_pred, loss = model(traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt)                        # forward         
    loss.backward()
    loss.unchain_backward()
    optimizer.update()

    return traj_pred, loss

def divide_chunks(l, n): 
    # split a list into chunks 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def adf_train_kSamples(parentInfo, kSamples, args):
    
    #args.num_epoch_cn = int(min(len(kSamples)/20 + 2, 30))
    for _ in range(args.num_epoch_cn):

        # Shuffle kSamples and splits them into batches
        if(len(kSamples) > args.batch_size_cn):
            random.shuffle(kSamples) #shuffle method
        listSamples =  list(divide_chunks(kSamples, args.batch_size_cn))

        for samples in listSamples:

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
                ego_in.to_gpu(args.gpu); traj_gt.to_gpu(args.gpu); scale_gt.to_gpu(args.gpu)

            traj_pred, loss = adf_train(parentInfo, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt)

    return np.asscalar(cuda.to_cpu(loss.data)) , parentInfo