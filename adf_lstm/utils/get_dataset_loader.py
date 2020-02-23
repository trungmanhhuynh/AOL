from utils.dataset import TrajectoryDataset
from torch.utils.data import DataLoader


def get_dataset_loader(args, data_file, flip=False,shuffle=False):

    dataset  = TrajectoryDataset(data_file = data_file, 
                                 obs_len = args.obs_len,
                                 pred_len = args.pred_len, 
                                 flip = flip)

    loader = DataLoader(dataset = dataset, 
                                 batch_size = args.batch_size,
                                 shuffle = shuffle, 
                                 num_workers=48)

    return dataset, loader 
