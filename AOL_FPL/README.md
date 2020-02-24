# AOL + FPL 


## Environment: 
  pytorch 1.0.0 

## Pre-processed data
#### FPL dataset
Download the processed data from FPL.

Run the following script:

```
python utils/generate_fpl_datasets.py /home/manhh/github/datasets/fpl_dataset/raw/id_list.txt --data_dir /home/manhh/github/datasets/fpl_dataset/raw/ --output_dir ./processed_data/fpl/id_list/ --eval_split 0 --traj_length 20 --traj_skip 2 --nb_splits 5 --seed 1701 --traj_skip_test 5 --debug --ego_type gridflow
```
(This script is modified from generate_datsets.py script from the original FPL's source code) 

Specify the following parameters:
  - directory (/home/manhh/github/datasets/fpl_dataset/raw/) for the fpl's data. 
  - id_list.txt if use 5 videos for testing or "id_list.txt" for full data 
  --eval_split : evaluation split (0->4)
  --output_dir : output data files 

#### ETH dataset.
Download the processed data and extract it to raw_data_json. 
```
gdown https://drive.google.com/uc?id=1AsNryRzah21DRlhga0sOM_o27MYyM5hh
tar -xvf raw_data_json.tar
```
Gerenerates train data:
```
python utils/generate_eth_datasets.py --raw_data_dir raw_data_json/ --output_dir processed_data/ --data_file utils/train_sequences.txt --mode train --traj_len 20 --ratio 0.8
```
Generates test data 
```
python utils/generate_eth_datasets.py --raw_data_dir raw_data_json/ --output_dir processed_data/eth/ --data_file utils/test_sequences.txt --mode test --traj_len 20 --slide 5
```

Specify the following parameters:
 --raw_data_dir: directory to raw_data_json 
 --data_file: video sequences used for training/testing. 
 --output_dir: output directory


## Train on FPL dataset
```
model=cnn && split=0 && python train.py --model $model --train_data processed_data/fpl/id_list/train_data_split_$split.joblib --val_data processed_data/fpl/id_list/val_data_split_"$split".joblib --save_root save/"$model"_gridflow/split_"$split" --obs_len 10 --pred_len 10 --gpu 0 --batch_size 64 --nepochs 100 --learning_rate 0.001 --resume_model ""
```

## ADF_test
#### FPL dataset
```
model=cnn_ego_scale && split=0 && python adf_test.py --model $model --val_data processed_data/fpl/id_list/val_data_split_$split.joblib --save_root save/$model/split_$split --resume_model save/$model/split_$split/models/model_epoch_100.pt --obs_len 10 --pred_len 10 --gpu 0 --batch_size 1 --num_past_chunks 128 --max_models 1
```
Important parameters:  
model: specify the model.  
--split: specify which split to test  
--resume_model: save model file  
--traj_file: specify if wanting to save trajectories  
  
Without Adaptation: specify --num_past_chunks 0 --max_models 1  
B-AOL: specify --num_past_chunks 1 --max_models 1  
AOL: specify --num_past_chunks 1 --max_models 10  


#### eth dataset.
```
model=cnn_ego_pose_scale && python adf_test.py --model $model --val_data processed_data/eth/test_data.joblib --save_root save/$model/eth --resume_model save/"$model"_gridflow/split_0/models/model_epoch_100.pt --obs_len 10 --pred_len 10 --gpu 1 --batch_size 1 --num_past_chunks 100 --max_models 10
```
Important parameters:  
model: specify the model.  
--split: specify which split to test  
--resume_model: save model file  
--traj_file: specify if wanting to save trajectories  
  
Without Adaptation: specify --num_past_chunks 0 --max_models 1  
B-AOL: specify --num_past_chunks 1 --max_models 1  
AOL: specify --num_past_chunks 1 --max_models 10  
