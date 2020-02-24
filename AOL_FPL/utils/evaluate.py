import numpy as np
from utils.dataset import std_denormalize, rel_to_nor
from chainer.backends import cuda

'''Calcuale ADE/FDE'''
def calculate_error(pred_data, target_data):

    temp = (pred_data - target_data)**2
    ADE = np.sqrt(temp[:,:,0] + temp[:,:,1])
    ADE = np.mean(ADE)

    FDE = np.sqrt(temp[:,-1,0] + temp[:,-1,1])
    FDE = np.mean(FDE)
    return ADE, FDE

def evaluate(traj_in, traj_gt, traj_pred, dataset):

        # convert relative to normalized data 
    traj_pred_nor = rel_to_nor(np.expand_dims(traj_in[:,-1,:], axis =1), traj_pred)
    traj_gt_nor = rel_to_nor(np.expand_dims(traj_in[:,-1,:], axis =1), traj_gt)

    # denormalize 
    traj_pred_abs = std_denormalize(traj_pred_nor, dataset.loc_mean, dataset.loc_var)
    pred_gt_abs = std_denormalize(traj_gt_nor, dataset.loc_mean, dataset.loc_var)
    ADE, FDE  = calculate_error(traj_pred_abs, pred_gt_abs)

    return ADE, FDE, traj_pred_abs
