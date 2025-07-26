#!/bin/bash
#SBATCH -p debug
#SBATCH -t 2-00:00:00
#SBATCH --nodelist=GPU1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --output=./shell_qaid/mujuco-idt-qaid-lsdycom3.out
#SBATCH --error=./shell_qaid/mujuco-idt-qaid-lsdycom3.err

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate qaid2

# export WANDB_MODE=offline 
# Function to run experiments
run_experiment() {
    local env=$1
    local expectile=$2
    local base_archs=("idt")
    local q_scale=$3
    local dc_embed_dim=$4
    local mlp_embed_dim=$5
    local dc_n_layer=$6
    local mlp_n_layer=$7
    local min_q=$8
    local conditioning=$9

    # IQL Pretrain
    # python3 main_iql_pretrain.py --env $env --expectile $expectile

    # QCS Experiments for each architecture
    for arch in "${base_archs[@]}"; do
        if [[ "$arch" == "dc" || "$arch" == "dt"||  "$arch" == "idt" ]]; then
            embed_dim=$dc_embed_dim
            n_layer=$dc_n_layer
        else
            embed_dim=$mlp_embed_dim
            n_layer=$mlp_n_layer
        fi
        python3 main_qcs.py --env $env --base_arch $arch --q_scale $q_scale --embed_dim $embed_dim --n_layer $n_layer ${min_q:+--min_q $min_q} ${conditioning:+--conditioning $conditioning} 
    done
}

# MuJoCo Experiments
run_experiment "hopper-expert-v2" 0.7 0.5 256 1024 4 3
# run_experiment "hopper-medium-v2" 0.7 0.5 256 1024 4 3
# run_experiment "walker2d-medium-v2" 0.7 1 256 1024 4 3 10
# run_experiment "walker2d-expert-v2" 0.7 1 256 1024 4 3 10
# run_experiment "halfcheetah-medium-v2" 0.7 1 256 1024 4 3
# run_experiment "halfcheetah-expert-v2" 0.7 1 256 1024 4 3

# # Antmaze Experiments
# run_experiment "antmaze-medium-play-v2" 0.9 0.2 512 1024 3 4 0 subgoal

# # Adroit Pen Experiments
# run_experiment "pen-human-v2" 0.7 0.01 128 256 3 3