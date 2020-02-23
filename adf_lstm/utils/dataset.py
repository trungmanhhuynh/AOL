import joblib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import KNNImputer


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

    # data_in ~  [batch_size, traj_len, 2]
    # start_pt ~ [batch_size,1,2] by using np.newaxis
    data_out = data_in +  start_pt

    return data_out

def  impute_missing_points(in_data):

    #in_data ~ [num_samples, traj_len, feature_Size]
    imputer =  KNNImputer(missing_values=0, n_neighbors=3)
    for sample_idx in range(in_data.shape[0]):
        in_data[sample_idx] = imputer.fit_transform(in_data[sample_idx])

    return in_data

def refine(in_data):
    print(in_data["gt_location"].shape)
    print(in_data["gt_location"][0])

    mask = np.ones(in_data["pose"].shape[0], dtype=bool)
    for sample_idx in range(in_data["pose"].shape[0]):
        for loc_idx in range(in_data["pose"].shape[1]): 

            RHip = in_data["pose"][sample_idx, loc_idx, 16:18]
            LHip = in_data["pose"][sample_idx, loc_idx, 22:24]
            if(np.count_nonzero(RHip) == 0 or np.count_nonzero(RHip) ==0):
                mask[sample_idx] = False

    in_data["start_frame"] = in_data["start_frame"][mask]
    in_data["seq_name"] = in_data["seq_name"][mask]
    in_data["pIds"] = in_data["pIds"][mask]
    in_data["pose"] = in_data["pose"][mask,:,:]
    in_data["openPoseLocation"] = in_data["openPoseLocation"][mask,:,:]
    in_data["scales"] = in_data["scales"][mask,:,:]
    in_data["gt_location"] = in_data["gt_location"][mask,:,:]
    in_data["egomotions"] = in_data["egomotions"][mask,:,:]

    print(in_data["gt_location"].shape)
    print(in_data["gt_location"][0])

    return in_data



class TrajectoryDataset(Dataset):
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
        print(self.data["pose"].shape)

        # parameters
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.flip = flip
        self.image_width = image_width

        # refine data
        #self.data= refine(self.data)

        # impute missing points
        #self.data["gt_location"] = impute_missing_points(self.data["gt_location"])
        #self.data["openPoseLocation"] = impute_missing_points(self.data["openPoseLocation"])
        #self.data["scales"] = impute_missing_points(self.data["scales"])
        #self.data["pose"] = impute_missing_points(self.data["pose"])

        self.loc_mean, self.loc_var = calc_mean_variance(self.data["gt_location"])     # location mean, var
       
       #self.scale_mean, self.scale_var = calc_mean_variance(self.data["scales"])   # scale mean, var
        #self.pose_mean, self.pose_var = calc_mean_variance(self.data["pose"])     # location mean, var
        #self.egomotions_mean, self.egomotions_var = calc_mean_variance(self.data["egomotions"])     # location mean, var

    def __len__(self):
        return self.data["gt_location"].shape[0]

    def __getitem__(self, idx):

        # using idx to find all trajs having the same start frame. 
        sample = {'traj_in': self.data["gt_location"][idx,:self.obs_len,:],                              # size [num_traj_in_frame, obs_len, 2]
                  'traj_gt': self.data["gt_location"][idx,self.obs_len:self.obs_len + self.pred_len,:], 
                  'traj_in_abs': self.data["gt_location"][idx,:self.obs_len,:], 
                  'traj_gt_abs': self.data["gt_location"][idx,self.obs_len:self.obs_len + self.pred_len,:], 
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
        sample['traj_gt'] = std_normalize(sample['traj_gt'], self.loc_mean, self.loc_var)
        
        # relative ground truth trajectory
        sample['traj_gt'] = nor_to_rel(sample['traj_in'][-1,:], sample['traj_gt'])

        return sample
