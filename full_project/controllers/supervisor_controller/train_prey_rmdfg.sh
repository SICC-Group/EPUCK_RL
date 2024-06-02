#!/bin/sh
env="Predator_prey"
num_agents=6  #9
num_stags=4 #6
num_factor=6  #18
algo="rddfg_cent_rw"
exp="debug"
name="shiyuchen"
seed_max=176
seed_min=176

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_min} ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train_prey.py --user_name ${name} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --num_agents ${num_agents} --num_factor ${num_factor} --seed ${seed} --num_stags ${num_stags} --episode_length 200 --use_soft_update \
    --num_env_steps 2000000 --n_training_threads 2 --cuda --eval_interval 20000 --num_eval_episodes 4 --batch_size 32 --use_reward_normalization \
    --miscapture_punishment 0 --reward_time 0 --highest_orders 2 --lr 1e-3 --adj_lr 1e-8 -train_interval_episode 4 --gamma 0.98 --entropy_coef 0 \
    --num_rank 1 --sparsity 0.3  --gain 0.01 --adj_begin_step 0 --gae_lambda 0.97
done
