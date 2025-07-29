#!/bin/bash
#SBATCH -p debug
#SBATCH -t 2-00:00:00
#SBATCH --nodelist=GPU1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --output=./shell_qaid/mujuco-idt-qaid-lstm-dycombine-new3.out
#SBATCH --error=./shell_qaid/mujuco-idt-qaid-lstm-dycombine-new3.err

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate qaid3

# export WANDB_MODE=offline 
# Function to run experiments
# run_experiment() {
#     local env=$1
#     local expectile=$2
#     local base_archs=("idt")
#     local q_scale=$3
#     local dc_embed_dim=$4
#     local mlp_embed_dim=$5
#     local dc_n_layer=$6
#     local mlp_n_layer=$7
#     local min_q=$8
#     local conditioning=$9

#     # IQL Pretrain
#     # python3 main_iql_pretrain.py --env $env --expectile $expectile

#     # QCS Experiments for each architecture
#     for arch in "${base_archs[@]}"; do
#         if [[ "$arch" == "dc" || "$arch" == "dt"||  "$arch" == "idt" || "$arch" == "idc" ]]; then
#             embed_dim=$dc_embed_dim
#             n_layer=$dc_n_layer
#         else
#             embed_dim=$mlp_embed_dim
#             n_layer=$mlp_n_layer
#         fi
#         python3 main_qcs.py --use_dpo --env $env --base_arch $arch --q_scale $q_scale --embed_dim $embed_dim --n_layer $n_layer ${min_q:+--min_q $min_q} ${conditioning:+--conditioning $conditioning} 
#         # python3 main_qcs.py --env $env --base_arch $arch --q_scale $q_scale --embed_dim $embed_dim --n_layer $n_layer ${min_q:+--min_q $min_q} ${conditioning:+--conditioning $conditioning} 
#     done
# }

# MuJoCo Experiments
# run_experiment "hopper-expert-v2" 0.7 0.5 256 1024 4 3
# run_experiment "hopper-medium-v2" 0.7 0.5 256 1024 4 3
# run_experiment "walker2d-medium-v2" 0.7 1 256 1024 4 3 10
# run_experiment "walker2d-expert-v2" 0.7 1 256 1024 4 3 10
# run_experiment "halfcheetah-medium-v2" 0.7 1 256 1024 4 3
# run_experiment "halfcheetah-expert-v2" 0.7 1 256 1024 4 3

# # Antmaze Experiments
# run_experiment "antmaze-medium-play-v2" 0.9 0.2 512 1024 3 4 0 subgoal

# # Adroit Pen Experiments
# run_experiment "pen-human-v2" 0.7 0.01 128 256 3 3


# python3 main_qcs.py \
#     --use_dpo \
#     --env hopper-expert-v2 \
#     --base_arch idt \
#     --q_scale 0.5 \
#     --embed_dim 256 \
#     --n_layer 4 \
#     --dpo_state_threshold 0.05 \
#     --dpo_action_diff_threshold 0.1 \
#     --dpo_beta 0.5 \
#     --dpo_reward_ratio 1.0 \
#     --dpo_model_path "./exp/old_models/hopper-expert-v2-idt-rtg-seed-0/model.pt" \
#     --save_model_name "dpo_model_st0_05_beta_0_05_rt1.0"

# python3 main_qcs.py \
#     --use_dpo \
#     --env hopper-expert-v2 \
#     --base_arch idt \
#     --q_scale 0.5 \
#     --embed_dim 256 \
#     --n_layer 4 \
#     --dpo_state_threshold 0.05 \
#     --dpo_action_diff_threshold 0.1 \
#     --dpo_beta 0.5 \
#     --dpo_reward_ratio 0.9 \
#     --dpo_model_path "./exp/old_models/hopper-expert-v2-idt-rtg-seed-0/model.pt" \
#     --save_model_name "dpo_model_st0_05_beta_0_05_rt0.9"


# python3 main_qcs.py \
#     --use_dpo \
#     --env hopper-expert-v2 \
#     --base_arch idt \
#     --q_scale 0.5 \
#     --embed_dim 256 \
#     --n_layer 4 \
#     --dpo_state_threshold 0.05 \
#     --dpo_action_diff_threshold 0.1 \
#     --dpo_beta 0.5 \
#     --dpo_reward_ratio 0.5 \
#     --dpo_model_path "./exp/old_models/hopper-expert-v2-idt-rtg-seed-0/model.pt" \
#     --save_model_name "dpo_model_st0_05_beta_0_05_rt0.5"

# python3 main_qcs.py \
#     --use_dpo \
#     --env hopper-expert-v2 \
#     --base_arch idt \
#     --q_scale 0.5 \
#     --embed_dim 256 \
#     --n_layer 4 \
#     --dpo_state_threshold 0.05 \
#     --dpo_action_diff_threshold 0.1 \
#     --dpo_beta 0.5 \
#     --dpo_reward_ratio 0.3 \
#     --dpo_model_path "./exp/old_models/hopper-expert-v2-idt-rtg-seed-0/model.pt" \
#     --save_model_name "dpo_model_st0_05_beta_0_05_rt0.3"

# 重新训练IDT-QAID 保存最优代码
# python3 main_qcs.py \
#     --env hopper-expert-v2 \
#     --base_arch idt \
#     --q_scale 0.5 \
#     --embed_dim 256 \
#     --n_layer 4 \
#     --save_model_name "hopper-expert-IDT-QAID-LSTMCombine-best"

# python3 main_qcs.py \
#     --env hopper-medium-v2 \
#     --base_arch idt \
#     --q_scale 0.5 \
#     --embed_dim 256 \
#     --n_layer 4 \
#     --save_model_name "hopper-medium-IDT-QAID-LSTMCombine-best"

# python3 main_qcs.py \
#     --env walker2d-medium-v2 \
#     --base_arch idt \
#     --q_scale 1 \
#     --embed_dim 256 \
#     --n_layer 4 \
#     --save_model_name "walker2d-medium-IDT-QAID-LSTMCombine-best"

# python3 main_qcs.py \
#     --env walker2d-expert-v2 \
#     --base_arch idt \
#     --q_scale 1 \
#     --embed_dim 256 \
#     --n_layer 4 \
#     --save_model_name "walker2d-expert-IDT-QAID-LSTMCombine-best"

python3 main_qcs.py \
    --env halfcheetah-medium-v2 \
    --base_arch idt \
    --q_scale 1 \
    --embed_dim 256 \
    --n_layer 4 \
    --save_model_name "halfcheetah-medium-IDT-QAID-LSTMCombine-best"

python3 main_qcs.py \
    --env halfcheetah-expert-v2 \
    --base_arch idt \
    --q_scale 1 \
    --embed_dim 256 \
    --n_layer 4 \
    --save_model_name "halfcheetah-expert-IDT-QAID-LSTMCombine-best"