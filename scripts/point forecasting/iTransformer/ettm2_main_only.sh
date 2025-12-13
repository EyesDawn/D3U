#!/bin/bash

### ETTm2 Dataset - Main Training Only ###
### Assumes pre-trained iTransformer checkpoint exists at: ###
### ./pretrain_checkpoints/iTransformer/all/ETTm2/192/checkpoint.pth ###

echo "========================================="
echo "ETTm2 Main Training (Diffusion Model)"
echo "Assuming iTransformer checkpoint exists"
echo "========================================="

## Data Configs ##
seq_len=96
label_len=48
pred_len=192
root_path='./dataset/ETT-small/'
data_path='ETTm2.csv'
dataset='ETTm2'
model_id_name='ETTm2'
random_seed=2021

## Model Configs ##
model_name='iTransformer'
enc_in=7
dec_in=7
c_out=7

## Main Training Parameters (Diffusion Model) ##
d_model_c=128
d_ff_main=128
e_layers_c=2
n_heads_c=8
d_model_d=128

## Training Configs ##
batch_size_main=128
test_batch_size_main=64

## GPU Config ##
gpu_id=2

## Log Paths ##
main_log="./logs/D3U/iTransformer/ETTm2_main.log"

# Create log directories if they don't exist
mkdir -p ./logs/D3U/iTransformer/

# Check if checkpoint exists
checkpoint_path="./pretrain_checkpoints/iTransformer/all/ETTm2/192/checkpoint.pth"
if [ ! -f "$checkpoint_path" ]; then
    echo "========================================="
    echo "ERROR: Pre-trained checkpoint not found!"
    echo "Expected at: $checkpoint_path"
    echo "Please run pre-training first or check the path."
    echo "========================================="
    exit 1
fi

echo "Found checkpoint at: $checkpoint_path"
echo ""
echo "========================================="
echo "Main Training (Diffusion Model) Starting..."
echo "========================================="

python -u main.py \
    --is_training \
    --seed $random_seed \
    --root_path $root_path \
    --data_path $data_path \
    --model_id ${model_id_name}_${seq_len}_${pred_len} \
    --model $model_name \
    --data $dataset \
    --data_name $model_id_name \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --e_layers_c $e_layers_c \
    --n_heads_c $n_heads_c \
    --d_model_c $d_model_c \
    --d_ff $d_ff_main \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --depth 1 \
    --d_model_d $d_model_d \
    --num_workers 4 \
    --itr 1 \
    --train_epochs 100 \
    --timesteps 100 \
    --batch_size $batch_size_main \
    --test_batch_size $test_batch_size_main \
    --des 'Exp' \
    --lradj 'type1' \
    --denoise_model 'PatchDN' \
    --kernel_size 15 \
    --fourier_factor 1.0 \
    --svq 1 \
    --wFFN 0 \
    --num_codebook 1 \
    --codebook_size 256 \
    --type_sample 'DPM_solver' \
    --DPMsolver_step 20 \
    --gpu $gpu_id \
    --parameterization "x_start" \
    --bias \
    2>&1 | tee -a $main_log

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Main Training Completed Successfully!"
    echo "========================================="
    echo "Log file: $main_log"
else
    echo "========================================="
    echo "ERROR: Main Training Failed!"
    echo "========================================="
    exit 1
fi

