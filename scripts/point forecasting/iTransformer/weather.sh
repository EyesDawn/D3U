#!/bin/bash

### Weather Dataset - Two-stage Training ###
### Stage 1: Pre-training with cond_model_main.py (train iTransformer) ###
### Stage 2: Main training with main.py (train diffusion model) ###

### Usage:
### ./weather.sh             # Run both stages (pre-training + main training)
### ./weather.sh main        # Skip pre-training, run only main training
### ./weather.sh pretrain    # Run only pre-training

# Parse command line argument
MODE=${1:-"both"}  # Default to "both" if no argument provided

echo "========================================="
echo "Starting Weather Training"
echo "Mode: $MODE"
if [ "$MODE" = "both" ]; then
    echo "Stage 1: Pre-training iTransformer (cond_model_main.py)"
    echo "Stage 2: Main training Diffusion Model (main.py)"
elif [ "$MODE" = "main" ]; then
    echo "Stage: Main training only (using existing pre-trained model)"
elif [ "$MODE" = "pretrain" ]; then
    echo "Stage: Pre-training only"
else
    echo "ERROR: Invalid mode '$MODE'"
    echo "Usage: $0 [both|main|pretrain]"
    exit 1
fi
echo "========================================="

## Data Configs ##
seq_len=96
label_len=48
pred_len=192
root_path='./dataset/weather/'
data_path='weather.csv'
dataset='custom'
model_id_name='weather'
random_seed=2021

## Model Configs ##
model_name='iTransformer'
enc_in=21
dec_in=21
c_out=21

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
gpu_id=4

## Log Paths ##
pretrain_log="./logs/iTrans_M_Weather_pretrain.log"
main_log="./logs/D3U/iTransformer/Weather_main.log"

# Create log directories if they don't exist
mkdir -p ./logs/D3U/iTransformer/
mkdir -p ./logs/

# Stage 1: Pre-training (only if mode is "both" or "pretrain")
if [ "$MODE" = "both" ] || [ "$MODE" = "pretrain" ]; then
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
    else
        echo "========================================="
        echo "ERROR: Stage 1 (Pre-training) Failed!"
        echo "========================================="
        exit 1
    fi
    
    # If mode is "pretrain", exit after pre-training
    if [ "$MODE" = "pretrain" ]; then
        echo "Pre-training completed. Exiting (mode=pretrain)."
        exit 0
    fi
fi

# Stage 2: Main Training (only if mode is "both" or "main")
if [ "$MODE" = "both" ] || [ "$MODE" = "main" ]; then
    echo ""
    echo "========================================="
    echo "Stage 2: Main Training (Diffusion Model) Starting..."
    echo "========================================="
    
    # Check if pre-trained model exists when mode is "main"
    if [ "$MODE" = "main" ]; then
        pretrained_model="./pretrain_checkpoints/$model_name/all/$dataset/$pred_len/checkpoint.pth"
        if [ ! -f "$pretrained_model" ]; then
            echo "ERROR: Pre-trained model not found at: $pretrained_model"
            echo "Please run pre-training first with: $0 pretrain"
            exit 1
        fi
        echo "Using existing pre-trained model: $pretrained_model"
    fi
    
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
        if [ "$MODE" = "both" ]; then
            echo "All training stages completed successfully!"
            echo "Pre-training log: $pretrain_log"
        fi
        echo "Main training log: $main_log"
    else
        echo "========================================="
        echo "ERROR: Stage 2 (Main Training) Failed!"
        echo "========================================="
        exit 1
    fi
fi

echo "========================================="
echo "Training Complete!"
echo "========================================="

