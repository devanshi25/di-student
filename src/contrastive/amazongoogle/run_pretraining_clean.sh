#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --export=NONE
MODEL=$1
CHECKPOINT=$2
BATCH=$3
LR=$4
TEMP=$5
AUG=$6
COMBO=$7
PROB=0.10
python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=amazon-google \
	--clean=True \
    --train_file /pfs/work7/workspace/scratch/ma_dmittal-dmittal/di-student/data/processed/amazon-google/contrastive/amazon-google-train.pkl.gz \
	--id_deduction_set /pfs/work7/workspace/scratch/ma_dmittal-dmittal/di-student/data/interim/amazon-google/amazon-google-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
	--combo=$COMBO \
    --output_dir /pfs/work7/workspace/scratch/ma_dmittal-dmittal/di-student/reports/contrastive/amazongoogle-clean-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-$TEMP-${MODEL##*/}/ \
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