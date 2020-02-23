import time
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from utils.logger import Logger
from utils.get_dataset_loader import get_dataset_loader
from utils.dataset import std_denormalize, rel_to_nor
from utils.evaluate import calculate_error
from config import get_args

from models.LSTM import LSTM

'''Read input arguments'''
args = get_args()                                  
logger = Logger(args, mode="test")                 # log 
logger.write("{}\n".format(args))
np.random.seed(seed=1701)

'''Create train/val datasets'''
val_dataset, val_loader = get_dataset_loader(args, args.test_data, flip=False, shuffle=False)

if __name__ == '__main__':

    # Create model 
    model = LSTM(obs_len=args.obs_len, pred_len=args.pred_len, use_cuda=args.use_cuda)
    if(args.use_cuda) : model = model.cuda() 
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.learning_rate)

    # Resume trained model 
    if(args.resume_model != ""):
        model.load_state_dict(torch.load(args.resume_model))
    
    model.eval()
    '''Train'''
    logger.write("Validation data size:{}".format(len(val_dataset)))
    start_time = time.time()
    vLoss, vADE, vFDE = 0, 0, 0
    for val_it, samples in enumerate(val_loader):      

        traj_in = Variable(samples["traj_in"].type(torch.FloatTensor))             # [batch_size, obs_len , 2]
        traj_gt = Variable(samples["traj_gt"].type(torch.FloatTensor))             # [batch_size, pred_len, 2]
        
        if(args.use_cuda): 
            traj_in, traj_gt =  traj_in.cuda(), traj_gt.cuda()

        traj_pred, loss = model(traj_in, traj_gt)                        # forward         

        '''Calculate ADE/FDE'''
        # convert relative to normalized data 
        traj_pred_nor = rel_to_nor(np.expand_dims(traj_in[:,-1,:].cpu().data.numpy(), axis =1), \
                                   traj_pred.cpu().data.numpy())
        traj_gt_nor = rel_to_nor(np.expand_dims(traj_in[:,-1,:].cpu().data.numpy(), axis =1), \
                                   traj_gt.cpu().data.numpy())

        # denormalize 
        traj_pred_abs = std_denormalize(traj_pred_nor, val_dataset.loc_mean, val_dataset.loc_var)
        pred_gt_abs = std_denormalize(traj_gt_nor, val_dataset.loc_mean, val_dataset.loc_var)
        ADE, FDE  = calculate_error(traj_pred_abs, pred_gt_abs)

        vLoss += loss.item()
        vADE += ADE.item()
        vFDE  += FDE.item()

    stop_time = time.time()
    logger.write("test_time={:.2f}   ".format((stop_time - start_time)*1000) +\
          "vLoss={:.5f}   ".format(vLoss/(val_it+1)) +\
          "vADE={:.2f}   ".format(vADE/(val_it+1)) +\
          "vFDE={:.2f}   ".format(vFDE/(val_it+1)) 
          )

