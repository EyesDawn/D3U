#!/bin/bash

### ETTm2 Dataset - Two-stage Training ###
### Stage 1: Pre-training with cond_model_main.py (train iTransformer) ###
### Stage 2: Main training with main.py (train diffusion model) ###

echo "========================================="
echo "Starting ETTm2 Two-Stage Training"
echo "Stage 1: Pre-training iTransformer (cond_model_main.py)"
echo "Stage 2: Main training Diffusion Model (main.py)"
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

## Stage 1: Pre-training Parameters (iTransformer) ##
d_model=128
d_ff=128
learning_rate=0.0001
lradj='type1'
delta=-0.0001

## Stage 2: Main Training Parameters (Diffusion Model) ##
d_model_c=128
d_ff_main=128
e_layers_c=2
n_heads_c=8
d_model_d=128

## Training Configs ##
batch_size_pretrain=16
test_batch_size_pretrain=1
batch_size_main=128
test_batch_size_main=64

## GPU Config ##
gpu_id=2

## Log Paths ##
pretrain_log="./logs/iTrans_M_ETTm2_pretrain.log"
main_log="./logs/D3U/iTransformer/ETTm2_main.log"

# Create log directories if they don't exist
mkdir -p ./logs/D3U/iTransformer/
mkdir -p ./logs/

echo "========================================="
echo "Stage 1: Pre-training iTransformer Starting..."
echo "========================================="

python -u cond_model_main.py \
        --is_training \
        --seed $random_seed \
        --checkpoints './checkpoints/all/' \
        --model $model_name \
        --model_id ${model_name}_${dataset}_${seq_len}_${pred_len} \
        --root_path $root_path \
        --data_path $data_path \
        --data $dataset \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --patch_len 16 \
        --stride 8 \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --e_layers 2 \
        --d_layers 1 \
        --train_epochs 10 \
        --patience 5 \
        --batch_size $batch_size_pretrain \
        --test_batch_size $test_batch_size_pretrain \
        --d_model $d_model \
        --d_ff $d_ff \
        --dropout 0.2 \
        --learning_rate $learning_rate \
        --lradj $lradj \
        --delta $delta \
        --gpu $gpu_id \
        2>&1 | tee -a $pretrain_log

# Check if pre-training was successful
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Stage 1: Pre-training Completed Successfully!"
    echo "========================================="
    echo ""
    echo "========================================="
    echo "Stage 2: Main Training (Diffusion Model) Starting..."
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
    
    # Check if main training was successful
    if [ $? -eq 0 ]; then
        echo "========================================="
        echo "Stage 2: Main Training Completed Successfully!"
        echo "========================================="
        echo ""
        echo "All training stages completed successfully!"
        echo "Pre-training log: $pretrain_log"
        echo "Main training log: $main_log"
    else
        echo "========================================="
        echo "ERROR: Stage 2 (Main Training) Failed!"
        echo "========================================="
        exit 1
    fi
else
    echo "========================================="
    echo "ERROR: Stage 1 (Pre-training) Failed!"
    echo "Skipping Stage 2..."
    echo "========================================="
    exit 1
fi
