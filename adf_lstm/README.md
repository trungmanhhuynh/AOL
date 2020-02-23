# adf_lstm 


# Enviroments: 
  pytorch 1.0.0 

## Gernerate training and testing data for First-person Locomotion (FPL) dataset

Required inputs: 
  - processed feature files in /home/manhh/github/datasets/fpl_dataset/raw/. The 
  - id_list.txt if use 5 videos for testing ;or "id_list.txt" for full data 

Remember to specify: 
  --eval_split 
  --output_dir 

Output: 
  - train_data/test_data stored in --output_dir

```
python utils/generate_fpl_datasets.py /home/manhh/github/datasets/fpl_dataset/raw/id_list.txt --data_dir /home/manhh/github/datasets/fpl_dataset/raw/ --output_dir ./processed_data/fpl/id_list/ --eval_split 0 --traj_length 20 --traj_skip 2 --nb_splits 5 --seed 1701 --traj_skip_test 5 --debug
```
This script is built upon the generate_datsets.py script from FPL's source code. 

## Gernerate training and testing data for other datasets.

Required inputs: 
 -- feature data stored in --raw_data_dir. To generate the feature data, look into 
each datasets for processing scripts. 

Remember to specify: 
  --ratio 
  --traj_len 

Output: 
  - train_data/test_data stored in --output_dir

```
python utils/generate_eth_datasets.py --raw_data_dir raw_data_json/ --output_dir processed_data/ --data_file utils/train_sequences.txt --mode train --traj_len 20 --ratio 0.8

python utils/generate_eth_datasets.py --raw_data_dir raw_data_json/ --output_dir processed_data/eth/ --data_file utils/test_sequences.txt --mode test --traj_len 20 --slide 5

```

## Train
```
CUDA_VISIBLE_DEVICES=3 && split=4 && python train.py --train_data processed_data/fpl/id_list/train_data_split_$split.joblib --val_data processed_data/fpl/id_list/val_data_split_$split.joblib --save_root save/LSTM/id_list/split$split --obs_len 10 --pred_len 10 --use_cuda --batch_size 64 --nepochs 100 --learning_rate 0.001
```

## Test 

split=0 && python test.py --test_data processed_data/fpl/id_list/val_data_split_$split.joblib --resume_model save/LSTM/id_list/split$split/models/model_epoch_100 --save_root save/LSTM/id_list/split$split --obs_len 10 --pred_len 10 --use_cuda --batch_size 64

## ADF_test
split=4 && CUDA_VISIBLE_DEVICES=3 python adf_test.py --test_data processed_data/fpl/id_list/val_data_split_$split.joblib --resume_model save/LSTM/id_list/split$split/models/model_epoch_100 --obs_len 10 --save_root save/LSTM/id_list/split$split --pred_len 10 --use_cuda --batch_size 1 --num_past_chunks 1 --max_models 1

Test eth datatset
```
CUDA_VISIBLE_DEVICES=3 python adf_test.py --test_data processed_data/eth/test_data.joblib --resume_model save/LSTM/id_list/split0/models/model_epoch_100 --obs_len 10 --save_root save/LSTM/eth/ --pred_len 10 --use_cuda --batch_size 1 --num_past_chunks 0 --max_models 1 --traj_file eth_traj.json
```

## plot trajectories on images
python utils/generate_image_result.py --res_file eth_traj.json --img_dir ~/github/datasets/eth_mobile/images/ --save_dir save/trajectories/
