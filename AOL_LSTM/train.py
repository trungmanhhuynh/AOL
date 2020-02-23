import time
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR


from utils.logger import Logger
from utils.get_dataset_loader import get_dataset_loader
from utils.dataset import std_denormalize, rel_to_nor
from utils.evaluate import calculate_error

from config import get_args
from models.LSTM import LSTM

'''Read input arguments'''
args = get_args()                                  
logger = Logger(args, mode="train")                 # log 
logger.write("{}\n".format(args))
np.random.seed(seed=1701)

'''Create train/val datasets'''
train_dataset, train_loader = get_dataset_loader(args, args.train_data, flip=True, shuffle=True)
val_dataset, val_loader = get_dataset_loader(args, args.val_data, flip=False, shuffle=False)

if __name__ == '__main__':

    # Create model 
    model = LSTM(obs_len=args.obs_len, pred_len=args.pred_len, use_cuda=args.use_cuda)
    if(args.use_cuda) : model = model.cuda() 
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.5)

    '''Train'''
    logger.write("Train data size:{}".format(len(train_dataset)))
    logger.write("Validation data size:{}".format(len(val_dataset)))
    
    best_ADE, best_epoch = 10000, 0

    for e in range(args.nepochs):

        start_time = time.time()
        eLoss, eADE, eFDE = 0, 0, 0
        scheduler.step()    # update learning rate
        model.train()
        for tr_it, samples in enumerate(train_loader):
            traj_in = Variable(samples["traj_in"].type(torch.FloatTensor))           # [batch_size, obs_len , 2]
            traj_gt = Variable(samples["traj_gt"].type(torch.FloatTensor))         
            
            if(args.use_cuda): 
                traj_in, traj_gt= traj_in.cuda(), traj_gt.cuda()

            optimizer.zero_grad()                                            # zero out gradients
            traj_pred, loss = model(traj_in, traj_gt)                        # forward         
            loss.backward()                                                  # backward       
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step() 
          
            '''Calculate ADE/FDE'''
            # convert relative to normalized data 
            traj_pred_nor = rel_to_nor(np.expand_dims(traj_in[:,-1,:].cpu().data.numpy(), axis =1), \
                                       traj_pred.cpu().data.numpy())
            traj_gt_nor = rel_to_nor(np.expand_dims(traj_in[:,-1,:].cpu().data.numpy(), axis =1), \
                                       traj_gt.cpu().data.numpy())

            # denormalize 
            traj_pred_abs = std_denormalize(traj_pred_nor, train_dataset.loc_mean, train_dataset.loc_var)
            pred_gt_abs = std_denormalize(traj_gt_nor, train_dataset.loc_mean, train_dataset.loc_var)

            ADE, FDE  = calculate_error(traj_pred_abs, pred_gt_abs)        

            eLoss += loss.item()
            eADE += ADE.item()
            eFDE += FDE.item()


        # validate each epoch
        vLoss, vADE, vFDE = 0, 0, 0
        model.eval()

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
        
        logger.write("epoch={}   ".format(e+1) +\
              "learning_rate={}   ".format(scheduler.get_lr()) +\
              "train_loss={:.5f}   ".format( eLoss/(tr_it+1)) +\
              "train_ADE={:.2f}   ".format(eADE/(tr_it+1)) +\
              "train_FDE={:.2f}   ".format(eFDE/(tr_it+1)) +\
              "train_time={:.2f}   ".format((stop_time - start_time)*1000) +\
              "vLoss={:.5f}   ".format(vLoss/(val_it+1)) +\
              "vADE={:.2f}   ".format(vADE/(val_it+1)) +\
              "vFDE={:.2f}   ".format(vFDE/(val_it+1)) 
              )

       

    # save last epoch model
    model_file = "{}/model_epoch_{}".format(args.save_model_dir, e + 1)
    torch.save(model.state_dict(), model_file)
