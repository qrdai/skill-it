#!/bin/bash
for SEED in 0 1 2 3 4
# k:            number of skills; the 0th dim of AM
# slicer:       the attribute name along which skills are defined
# mw:           Use 'multiplicative weights' (e.g., Skill-It online sampling) algorithm
# eta:          (dynamic) learning rate "yi-ta" in skill-it proportions update rule
# max_steps:    Maximum number of steps to train for.
# update_steps: How often to update multiplicative weights. NOTE: T = max_steps / update_steps, is the total # of updating rounds
# num_ckpts:    Number of checkpoints to evaluate the model at.

# n_epochs:     default to 1 (overriden by max_steps)
# lr:           default to 5e-5
# batch_size:   default to 4 (for gpt-neo-125M)
do 
    python3 main.py \
        --task_name ni \
        --train_data_dir /dfs/scratch0/mfchen/natural-instructions \
	    --val_data_dir /dfs/scratch0/mfchen/natural-instructions \
        --dev_split_path ./aux_data/xlingual_dev_split_map.pkl \
        --ni_task_info_path ./aux_data/ni_xlingual_task_info.pkl \
        --selection_seed ${SEED} \
        --max_steps 600 \
        --xlingual \
        --slice_list question_answering english english question_answering spanish spanish question_generation english english question_generation spanish spanish \
        --k 4 \
        --slicer task_category input_language output_language \
        --sample_rule mixture \
        --target_mask 0 0 0 1 \
        --mw \
        --eta 0.8 \
        --mw_window 3 \
        --update_steps 100 \
        --graph_path ./ni_graphs/spanish_qg_normalized.npy \
        --filter_val_skills \
        --num_ckpts 6
done 
