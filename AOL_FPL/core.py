from models.cnn import CNN, CNN_ego_scale, CNN_scale, CNN_pose_scale, CNN_ego_pose_scale, CNN_ego
import sys
def create_model(args):

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

    if(args.gpu >=0) : model.to_gpu(args.gpu)
    return model