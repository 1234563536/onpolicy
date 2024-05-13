#!/bin/sh
env="Market"
scenario="simple_speaker_listener"
num_landmarks=3
num_agents=119
algo="mappo" # "ippo""rmappo"
exp="check"
seed_max=1



echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../../../train_market.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 10 --n_rollout_threads 10 --num_mini_batch 64 --episode_length 24 --num_env_steps 20000 \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "xxx" --user_name "huke" --run_tag 6 --eval_interval 1 \
    --share_policy --use_gae False --value_norm True
done
sleep 10000