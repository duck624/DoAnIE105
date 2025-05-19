#!/bin/bash
# Federated training and attack measurement calculating command
dataset=mnist
model_name=alexnet
opt=sgd
seed=1
lr=0.1
local_epoch=1
num_users=5
samples_per_user=500  # Đã giảm từ 1500 xuống 500
batch_size=32  # Đã tăng từ 1 lên 32
epochs=70  # Tăng từ 10 lên 70

# IID experiment (Training)
save_dir_iid=log_fedmia/iid/mnist_K${num_users}_N${samples_per_user}_alexnet_s${seed}_iid1
mkdir -p "$save_dir_iid"
python3 main.py --seed $seed --num_users $num_users --iid 1 \
--dataset $dataset --model_name $model_name --epochs $epochs --local_ep $local_epoch \
--lr $lr --batch_size $batch_size --optim $opt --save_dir "$save_dir_iid" --log_folder_name "$save_dir_iid" \
--lr_up cosine --MIA_mode 1 --device cpu --samples_per_user $samples_per_user &> "${save_dir_iid}/raw_logs"

# Non-IID experiment (Training)
save_dir_noniid=log_fedmia/noniid/mnist_K${num_users}_N${samples_per_user}_alexnet_s${seed}_noniid1
mkdir -p "$save_dir_noniid"
python3 main.py --seed $seed --num_users $num_users --iid 0 --beta 1.0 \
--dataset $dataset --model_name $model_name --epochs $epochs --local_ep $local_epoch \
--lr $lr --batch_size $batch_size --optim $opt --save_dir "$save_dir_noniid" --log_folder_name "$save_dir_noniid" \
--lr_up cosine --MIA_mode 1 --device cpu --samples_per_user $samples_per_user 2>&1 | tee "${save_dir_noniid}/raw_logs"
