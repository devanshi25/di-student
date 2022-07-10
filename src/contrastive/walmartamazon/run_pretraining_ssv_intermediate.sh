#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --export=NONE
MODEL=$1
CHECKPOINT=$2
BATCH=$3
LR=$4
TEMP=$5
INTERM=$6
AUG=$7
python run_pretraining_ssv.py \
    --do_train \
	--dataset_name=walmart-amazon \
    --train_file /pfs/work7/workspace/scratch/ma_rpeeters-ma_rpeeters-0/phd/dev/di-research/data/processed/walmart-amazon/contrastive/walmart-amazon-train.pkl.gz \
	--id_deduction_set /pfs/work7/workspace/scratch/ma_rpeeters-ma_rpeeters-0/phd/dev/di-research/data/interim/walmart-amazon/walmart-amazon-train.json.gz \
	--interm_file /pfs/work7/workspace/scratch/ma_rpeeters-ma_rpeeters-0/phd/dev/di-research/data/processed/wdc-lspc/contrastive/pre-train/intermediate/$INTERM.pkl.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
    --output_dir /pfs/work7/workspace/scratch/ma_rpeeters-ma_rpeeters-0/phd/dev/di-research/reports/contrastive/walmartamazon-ssv-$AUG$BATCH-$LR-$TEMP-$INTERM-${MODEL##*/}/ \
	--temperature=$TEMP \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=200 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--logging_strategy="epoch" \
	--augment=$AUG \