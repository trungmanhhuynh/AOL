import argparse
import os
import sys 

def get_args():

    parser = argparse.ArgumentParser()

    # Hardly change 
    parser.add_argument('--save_freq', type=int, default= 1, help='')
    parser.add_argument('--info_freq', type=int, default=10, help='frequency to print out')
    parser.add_argument('--dropout', type=float, default= 0.1, help='probability of keeping neuron during dropout')
    parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients to this magnitude')
    parser.add_argument('--optim', type=str, default='Adam', help="ctype of optimizer: 'rmsprop' 'adam'")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay', type=float, default=0.95, help='decay rate for rmsprop')
    parser.add_argument('--momentum', type=float, default=1, help='momentum for rmsprop')

    # model
    parser.add_argument('--model', type=str, default="cnn", help ="cnn, cnn_scale, cnn_scale_pose, cnn_scale_ego, cnn_scale_pose_ego")
    parser.add_argument('--inter_list', type=int, nargs="*", default=[256])
    parser.add_argument('--last_list', type=int, nargs="*", default=[])
    parser.add_argument('--channel_list', type=int, nargs="*", default=[32, 64, 128, 128])
    parser.add_argument('--deconv_list', type=int, nargs="*", default=[256, 128, 64, 32])
    parser.add_argument('--ksize_list', type=int, nargs="*", default=[3, 3, 3, 3])
    parser.add_argument('--dc_ksize_list', type=int, nargs="*", default=[])
    parser.add_argument('--pad_list', type=int, nargs="*", default=[])
    parser.add_argument('--lr_step_list', type=float, nargs="*", default=[5000,10000,15000])

    # Often changes by using command 
    parser.add_argument('--obs_len', type=int, default= 10, help='number of obseved frames')
    parser.add_argument('--pred_len', type=int, default= 10, help='number of predicted frames')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--nepochs', type=int, default= 300, help='number of epochs')
    parser.add_argument('--chunk_size', type=int, default= 20, help='chunk of frames')
    parser.add_argument('--chunk_slide', type=int, default= 1, help='sliding of frames')
    parser.add_argument('--gpu', type=int, default=0, help='gpu number')

    #check these before running
    parser.add_argument('--train_data', type=str, help='file used for training')
    parser.add_argument('--val_data', type=str, help='file used for validation')
    parser.add_argument('--test_data', type=str, help='file used for testing')
    parser.add_argument('--save_root', type=str, help='save root')
    parser.add_argument('--resume_model', type=str, default ="", help='resume_model')
    parser.add_argument('--traj_file', type=str, default ="", help='result trajectory file')

    parser.add_argument('--replacement_policy', type=str, default= 'LRU', help='network replacement policy: None,LRU')
    parser.add_argument('--num_past_chunks', type=int, default= 1, help='number of past chunks')
    parser.add_argument('--num_epoch_cn', type=int, default= 3, help='number of epoch for each continous learning step')
    parser.add_argument('--max_models', type=int, default= 1, help='maximum number of networks')
    parser.add_argument('--batch_size_cn', type=int, default= 64, help='number of epoch for each continous learning step')
    parser.add_argument('--epsilon', type=float, default= 0, help='used in hybrid')


    args = parser.parse_args()

    args.log_dir = os.path.join(args.save_root, 'logs')
    args.save_model_dir =  os.path.join(args.save_root, 'models')
    args.traj_len = args.pred_len + args.obs_len


    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    return args