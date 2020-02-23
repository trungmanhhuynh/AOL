import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file', type=str, help='result file')
    parser.add_argument('--save_dir', type=str, help='where to save res images')
    parser.add_argument('--img_dir', type=str, help='where to save res images')
    parser.add_argument('--obs_len', type=int, default=10, help='obs len')
    parser.add_argument('--pred_len', type=int, default=10, help='pred_len')
    args = parser.parse_args()

    traj_len = args.obs_len + args.pred_len
    '''Read res file'''
    with open(args.res_file, "r") as f:
        res_list = json.load(f)


    for sample in res_list:
        seq_name = sample["seq_name"][0]
        pid = sample["pId"][0]
        start_frame = int(sample["start_frame"][0])
        traj_in_abs = np.asarray(sample["traj_in_abs"])
        traj_gt_abs = np.asarray(sample["traj_gt_abs"])
        pred_traj = np.asarray(sample["pred_traj"])


        frame_fn = os.path.join(args.img_dir, seq_name, "{:06d}.jpg".format(start_frame + 9))
        print(frame_fn)
        if not os.path.exists(frame_fn):
            print("image file not found") 
            exit(1) 

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        outfile = os.path.join(args.save_dir, seq_name+"_{:06d}_{}.jpg".format(start_frame + 9, pid))  


        img = plt.imread(frame_fn)
        fig, ax = plt.subplots(nrows=1, ncols=1)

        #print(traj_list)
        ax.plot(traj_in_abs[0,:,0], 960 - traj_in_abs[0,:,1], '-', linewidth=3, color='yellow')
        ax.plot(traj_gt_abs[0,:,0], 960 - traj_gt_abs[0,:,1], '-', linewidth=3, color='red')
        ax.plot(pred_traj[0,:,0],  960 - pred_traj[0,:,1], '-', linewidth=3, color='blue')
        ax.plot(pred_traj[0,-1,0], 960 - pred_traj[0,-1,1], 'o', linewidth=50, color='blue')        # final location

        
        #ax.invert_yaxis()
        ax.imshow(img, extent=[0, 1280, 0, 960])
        #ax.text(0, 100, "ADE={:.2f}".format(ADE_list[1]), fontsize= 15, color='red')
        #ax.text(10, 10, "FDE={:.2f}".format(FDE_list[1]), fontsize= 15, color='red')

        fig.savefig(outfile)
        plt.close()
        print("Write results on image: {}".format(outfile))
