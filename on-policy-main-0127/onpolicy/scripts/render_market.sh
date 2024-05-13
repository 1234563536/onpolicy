#!/bin/sh
env="Market"
scenario="simple_speaker_listener"
num_landmarks=3
num_agents=118
algo="mappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_market.py --save_gifs False --share_policy False --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render False --episode_length 25 --render_episodes 5 \
    --model_dir "/home/ouazusakou/New_disk/on-policy-main/onpolicy/scripts/results/Market/simple_speaker_listener/mappo/check/run17/models"
done
