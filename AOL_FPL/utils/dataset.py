import joblib
import numpy as np
from chainer.dataset import dataset_mixin

def calc_mean_variance(data):

    #data shape must be [num_traj, traj_len, N], where N is feature size
    data = np.concatenate(data, axis = 0)  # (num_traj*traj_len, N)
    mean = data.mean(axis = 0)
    var = data.std(axis = 0)

    return mean, var

def std_normalize(data_in, mean, var):

    # data_in ~ [num_ped, traj_len, 2]
    # mean, var ~ [1,2]
    data_out = (data_in - mean)/var
    return data_out


def std_denormalize(data_in, mean, var):

    # data_in ~ [num_ped, traj_len, 2]
    # mean, var ~ [1,2]
    data_out = data_in*var + mean
    return data_out

def normalize_pose(pose, scale):

    # scale ~ # [num_person, traj_len, 1]
    # pose ~ # [num_person, traj_len, 25*2]

    mid_hip = pose[:,:,16:18]                  # [num_person, traj_len, 2]

    pose[:,:,0::2] = (pose[:,:,0::2] -  np.expand_dims(mid_hip[:,:,0], axis=2))/scale          # for x coordinates of joints
    pose[:,:,1::2] = (pose[:,:,1::2] -  np.expand_dims(mid_hip[:,:,1], axis=2))/scale          # for y coordinates of joints
    return pose


def nor_to_rel(start_pt, data_in):

    # data_in ~  [traj_len, 2]
    # start_pt ~ [1,2] 
    data_out = data_in -  start_pt

    return data_out

def rel_to_nor(start_pt, data_in):

    # data_in ~  [batch_size, pred_len, 2]
    # start_pt ~ [batch_size,1,2] 
    data_out = data_in +  start_pt

    return data_out



class TrajectoryDataset(dataset_mixin.DatasetMixin):
    """ Trajectory Dataset 
        data is dictionary of features. 
            "start_frame" - ~ [num_samples]
            "seq_name" - ~ [num_samples]
            "pIds" - ~ [num_samples]
            "pose" -   [num_samples, traj_len, 18, 2]                       
            "openPoseLocation" - [num_samples, 20 ,2 ]
            "scales" - [num_samples, traj_len, 1]
            "gt_location" - [num_samples, 20 ,2 ]
            "egomotions" - [num_samples, 20 ,24]
    """
    def __init__(self, data_file, obs_len=10, pred_len=10, flip=False, \
                 image_width=1280):

        self.data = joblib.load(data_file)       # read data (train, val, test)
        self.data["pose"] = self.data["pose"].reshape(self.data["pose"].shape[0],self.data["pose"].shape[1], 36)
        # self.data["pose"].shape now is ~  [num_samples, traj_len, 36] 

        # parameters
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.flip = flip
        self.image_width = image_width

        #self.loc_mean = np.array([640., 476.23620605])
        #self.loc_var = np.array([227.59802246, 65.00177002])

        self.loc_mean, self.loc_var = calc_mean_variance(self.data["gt_location"])     # location mean, var
        self.scale_mean, self.scale_var = calc_mean_variance(self.data["scales"])   # scale mean, var
        self.pose_mean, self.pose_var = calc_mean_variance(self.data["pose"])     # location mean, var
        self.ego_mean, self.ego_var = calc_mean_variance(self.data["egomotions"])     # location mean, var

    def __len__(self):
        return self.data["gt_location"].shape[0]

    def get_example(self, idx):

        # using idx to find all trajs having the same start frame. 
        sample = {'traj_in': self.data["gt_location"][idx,:self.obs_len,:], 
                  'traj_in_abs': self.data["gt_location"][idx,:self.obs_len,:],                             # size [num_traj_in_frame, obs_len, 2]
                  'scale_in': self.data["scales"][idx,:self.obs_len,:],  
                  'pose_in': self.data["pose"][idx,:self.obs_len,:], 
                  'ego_in': self.data["egomotions"][idx,:self.obs_len,:], 
                  'traj_gt': self.data["gt_location"][idx,self.obs_len:self.obs_len + self.pred_len,:], 
                  'traj_gt_abs': self.data["gt_location"][idx,self.obs_len:self.obs_len + self.pred_len,:], 
                  'scale_gt': self.data["scales"][idx,self.obs_len:self.obs_len + self.pred_len,:],
                  'start_frame'  : self.data['start_frame'][idx],
                  'seq_name': self.data['seq_name'][idx], 
                  'pId': self.data['pIds'][idx]}

        # horizontal_flip a sample  
        horizontal_flip = np.random.random() < 0.5 if self.flip else False
        if(horizontal_flip):
            sample['traj_in'][:,0] = self.image_width  - sample['traj_in'][:,0]
            sample['traj_gt'][:,0] = self.image_width  - sample['traj_gt'][:,0]

        # normalize data
        sample['traj_in'] = std_normalize(sample['traj_in'], self.loc_mean, self.loc_var)
        sample['scale_in'] = std_normalize(sample['scale_in'], self.scale_mean, self.scale_var)
        sample['pose_in'] = std_normalize(sample['pose_in'], self.pose_mean, self.pose_var)
        sample['ego_in'] = std_normalize(sample['ego_in'], self.ego_mean, self.ego_var)
        sample['scale_gt'] = std_normalize(sample['scale_gt'], self.scale_mean, self.scale_var)
        sample['traj_gt'] = std_normalize(sample['traj_gt'], self.loc_mean, self.loc_var)


        # relative ground truth trajectory
        sample['traj_gt'] = nor_to_rel(sample['traj_in'][-1,:], sample['traj_gt'])
        sample['scale_gt'] = nor_to_rel(sample['scale_in'][-1,:], sample['scale_gt'])

        return sample
