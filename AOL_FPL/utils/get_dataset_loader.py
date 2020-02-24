from utils.dataset import TrajectoryDataset
from chainer import iterators


def get_dataset_loader(args, data_file, flip=False,shuffle=False):

    dataset  = TrajectoryDataset(data_file = data_file, 
                                 obs_len = args.obs_len,
                                 pred_len = args.pred_len, 
                                 flip = flip)

    loader = iterators.MultiprocessIterator(dataset,
                                           args.batch_size,
                                           repeat = False,  
                                           shuffle = shuffle,
                                           n_processes=48)

    return dataset, loader 
