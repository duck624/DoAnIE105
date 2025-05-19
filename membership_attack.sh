#!/bin/bash
# Membership inference attack command

# Common parameters
seed=1
total_epoch=70
gpu=0
num_users=5
samples_per_user=1500

# Activate virtual environment (if exists)
if [ -f /home/aiduc/FedMIA/venv/bin/activate ]; then
    source /home/aiduc/FedMIA/venv/bin/activate || { echo "Error: Cannot activate virtual environment"; exit 1; }
else
    echo "Warning: Virtual environment not found, using system Python."
fi

# IID attack
path_iid="/home/aiduc/FedMIA/log_fedmia/iid/mnist_K${num_users}_N${samples_per_user}_alexnet_s${seed}_iid1/mnist_K${num_users}_N${samples_per_user}_alexnet_s${seed}_iid1/alexnet/mnist"
save_dir_iid="/home/aiduc/FedMIA/log_fedmia/iid/mnist_K${num_users}_N${samples_per_user}_alexnet_s${seed}_iid1"
mkdir -p "$save_dir_iid" || { echo "Error: Cannot create $save_dir_iid"; exit 1; }
echo "Running attack with path_iid=$path_iid, save_dir=$save_dir_iid, epoch=$total_epoch, gpu=$gpu, seed=$seed"
if [ ! -f /home/aiduc/FedMIA/mia_attack_auto.py ]; then
    echo "Error: mia_attack_auto.py not found in /home/aiduc/FedMIA"
    exit 1
fi
if [ ! -d "$path_iid" ]; then
    echo "Error: Directory $path_iid does not exist"
    exit 1
fi
python -u /home/aiduc/FedMIA/mia_attack_auto.py "$path_iid" "$total_epoch" "$gpu" "$seed" 2>&1 | tee -a "$save_dir_iid/attack_log.txt"
if [ $? -ne 0 ]; then
    echo "Error: Failed to run mia_attack_auto.py. Check $save_dir_iid/attack_log.txt for details."
    exit 1
fi
echo "Attack completed. Check $save_dir_iid/attack_log.txt for results."

# Non-IID attack (uncomment if needed)
# path_noniid="/home/aiduc/FedMIA/log_fedmia/noniid/mnist_K${num_users}_N${samples_per_user}_alexnet_s${seed}_noniid1/mnist_K${num_users}_N${samples_per_user}_alexnet_s${seed}_noniid1/alexnet/mnist"
# save_dir_noniid="/home/aiduc/FedMIA/log_fedmia/noniid/mnist_K${num_users}_N${samples_per_user}_alexnet_s${seed}_noniid1"
# mkdir -p "$save_dir_noniid" || { echo "Error: Cannot create $save_dir_noniid"; exit 1; }
# echo "Running non-IID attack with path_noniid=$path_noniid, save_dir=$save_dir_noniid, epoch=$total_epoch, gpu=$gpu, seed=$seed"
# if [ ! -d "$path_noniid" ]; then
#     echo "Error: Directory $path_noniid does not exist"
#     exit 1
# fi
# python -u /home/aiduc/FedMIA/mia_attack_auto.py "$path_noniid" "$total_epoch" "$gpu" "$seed" 2>&1 | tee -a "$save_dir_noniid/attack_log.txt"
# if [ $? -ne 0 ]; then
#     echo "Error: Failed to run mia_attack_auto.py for non-IID. Check $save_dir_noniid/attack_log.txt for details."
#     exit 1
# fi
# echo "Non-IID attack completed. Check $save_dir_noniid/attack_log.txt for results."
