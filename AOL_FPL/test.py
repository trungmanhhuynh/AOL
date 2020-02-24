import time
import numpy as np
import cupy as cp
import chainer 
from chainer import optimizers, Variable, cuda, serializers

from utils.logger import Logger
from utils.get_dataset_loader import get_dataset_loader
from utils.dataset import std_denormalize, rel_to_nor
from utils.evaluate import calculate_error, evaluate

from config import get_args
from core import create_model

'''Read input arguments'''
args = get_args()                                  
logger = Logger(args, mode="train")                 # log 
logger.write("{}\n".format(args))
np.random.seed(seed=1701)

'''Create train/val datasets'''
val_dataset, val_loader = get_dataset_loader(args, args.val_data, flip=False, shuffle=False)
logger.write("Validation data size:{}".format(len(val_dataset)))

if __name__ == '__main__':

    # Create model 
    model = create_model(args)

    # Load model 
    if args.resume_model != "":
        serializers.load_npz(args.resume_model, model)


    # validate each epoch
    vLoss, vADE, vFDE = 0, 0, 0
    chainer.config.train = False
    chainer.config.enable_backprop = False
    val_loader.reset()
    for val_it, samples in enumerate(val_loader):      

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

        # test
        #traj_pred, loss = model(traj_in, scale_in, pose_in, ego_in, traj_gt)                        # forward         
        traj_pred, loss = model(traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt)                        # forward         

        #Calculate ade/fde
        ADE, FDE = evaluate(cuda.to_cpu(traj_in.data), cuda.to_cpu(traj_gt.data), cuda.to_cpu(traj_pred.data), val_dataset)
        vLoss += np.asscalar(cuda.to_cpu(loss.data)) 
        vADE += ADE
        vFDE += FDE

    stop_time = time.time()       
    logger.write(
          "vLoss={:.5f}   ".format(vLoss/(val_it+1)) +\
          "vADE={:.2f}   ".format(vADE/(val_it+1)) +\
          "vFDE={:.2f}   ".format(vFDE/(val_it+1)) 
          )