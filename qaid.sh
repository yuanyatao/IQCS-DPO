#!/bin/bash
#SBATCH -p debug
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --output=./shell_qaid/hopper-medium-dc.out
#SBATCH --error=./shell_qaid/hopper-medium-dc.err

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate qaid

# run_experiment() {
#     local env=$1
#     local expectile=$2
#     local base_archs=("dc" "mlp")
#     local q_scale=$3
#     local dc_embed_dim=$4
#     local mlp_embed_dim=$5
#     local dc_n_layer=$6
#     local mlp_n_layer=$7
#     local min_q=$8
#     local conditioning=$9

#     # IQL Pretrain
#     python3 main_iql_pretrain.py --env $env --expectile $expectile

#     # QCS Experiments for each architecture
#     for arch in "${base_archs[@]}"; do
#         if [ "$arch" == "dc" ]; then
#             embed_dim=$dc_embed_dim
#             n_layer=$dc_n_layer
#         else
#             embed_dim=$mlp_embed_dim
#             n_layer=$mlp_n_layer
#         fi
#         python3 main_qcs.py --env $env --base_arch $arch --q_scale $q_scale --embed_dim $embed_dim --n_layer $n_layer ${min_q:+--min_q $min_q} ${conditioning:+--conditioning $conditioning} 
#     done
# }

# run_experiment "hopper-medium-v2" 0.7 0.5 256 1024 4 3

python3 main_qcs.py --env hopper-medium-v2 --base_arch dc --q_scale 0.5 --embed_dim 256 --n_layer 4

echo "作业结束时间：$(date)"