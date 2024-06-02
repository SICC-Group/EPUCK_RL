import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description="WEBOTS-RL", formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="rmatd3", choices=[
                        "rmaddpg", "qtran","qplex","wqmix","qmix", "vdn", "matd3", "maddpg", "masac", "mqmix", "mvdn","rmfg_cent","rddfg_cent_rw","sopcg","casec"])
    parser.add_argument("--experiment_name", type=str, default="check")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True)
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True)
    parser.add_argument('--n_training_threads', type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument('--n_rollout_threads', type=int,  default=1,
                        help="Number of parallel envs for training rollout")
    parser.add_argument('--n_eval_rollout_threads', type=int,  default=1,
                        help="Number of parallel envs for evaluating rollout")
    parser.add_argument('--num_env_steps', type=int,
                        default=2000000, help="Number of env steps to train for")
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help="Whether to use weights&biases, if not, use tensorboardX instead")
    parser.add_argument('--user_name', type=str, default="syc")

    # env parameters
    parser.add_argument('--env_name', type=str, default="StarCraft2")

    # replay buffer parameters
    parser.add_argument('--episode_length', type=int,
                        default=80, help="Max length for any episode")
    parser.add_argument('--buffer_size', type=int, default=5000,
                        help="Max # of transitions that replay buffer can contain")
    parser.add_argument('--adj_buffer_size', type=int, default=4,
                        help="Max # of transitions that adj replay buffer can contain")
    parser.add_argument('--use_reward_normalization', action='store_true',
                        default=False, help="Whether to normalize rewards in replay buffer")
                        
    # prioritized experience replay
    parser.add_argument('--use_per', action='store_true', default=False,
                        help="Whether to use prioritized experience replay")
    parser.add_argument('--per_nu', type=float, default=0.9,
                        help="Weight of max TD error in formation of PER weights")
    parser.add_argument('--per_alpha', type=float, default=0.6,
                        help="Alpha term for prioritized experience replay")
    parser.add_argument('--per_eps', type=float, default=1e-6,
                        help="Eps term for prioritized experience replay")
    parser.add_argument('--per_beta_start', type=float, default=0.4,
                        help="Starting beta term for prioritized experience replay")

    # network parameters  
    parser.add_argument("--use_centralized_Q", action='store_false',
                        default=True, help="Whether to use centralized Q function")
    parser.add_argument('--share_policy', action='store_false',
                        default=True, help="Whether agents share the same policy")
    parser.add_argument('--hidden_size', type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument('--layer_N', type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument('--use_ReLU', action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument('--use_feature_normalization', action='store_true',
                        default=False, help="Whether to apply layernorm to the inputs")
    parser.add_argument('--use_orthogonal', action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    parser.add_argument("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_cell", action='store_true',
                        default=False, help="Whether to use GRUCell")

    # recurrent parameters
    parser.add_argument('--prev_act_inp', action='store_true', default=False,
                        help="Whether the actor input takes in previous actions as part of its input")
    parser.add_argument("--use_rnn_layer", action='store_false',
                        default=True, help='Whether to use a recurrent policy')
    parser.add_argument("--use_naive_recurrent_policy", action='store_false',
                        default=True, help='Whether to use a naive recurrent policy')
    # TODO now only 1 is support
    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument('--data_chunk_length', type=int, default=80,
                        help="Time length of chunks used to train via BPTT")
    parser.add_argument('--burn_in_time', type=int, default=0,
                        help="Length of burn in time for RNN training, see R2D2 paper")

    # attn parameters
    parser.add_argument("--attn", action='store_true', default=False)
    parser.add_argument("--attn_N", type=int, default=1)
    parser.add_argument("--attn_size", type=int, default=64)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_average_pool",
                        action='store_false', default=True)
    parser.add_argument("--use_cat_self", action='store_false', default=True)

    # optimizer parameters
    parser.add_argument('--adj_lr', type=float, default=5e-2,
                        help="Learning rate for Adam")
    parser.add_argument('--lr', type=float, default=5e-4,
                        help="Learning rate for Adam")
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--opti_alpha", type=float, default=0.99,
                        help='RMSProp alpha')
    parser.add_argument("--weight_decay", type=float, default=0)

    # algo common parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of buffer transitions to train on at once")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor for env")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True)
    parser.add_argument("--max_grad_norm", type=float, default=10,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--use_huber_loss', action='store_true',
                        default=False, help="Whether to use Huber loss for critic update")
    parser.add_argument("--huber_delta", type=float, default=10.0)

    # soft update parameters
    parser.add_argument('--use_soft_update', action='store_false',
                        default=True, help="Whether to use soft update")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="Polyak update rate")
    # hard update parameters
    parser.add_argument('--hard_update_interval_episode', type=int, default=200,
                        help="After how many episodes the lagging target should be updated")
    parser.add_argument('--hard_update_interval', type=int, default=200,
                        help="After how many timesteps the lagging target should be updated")

    # dcg and ddfg parameters
    #     policy network parameters
    parser.add_argument("--use_dyn_graph", action='store_false',
                        default=True, help="Whether to use Generative graph network")
    parser.add_argument("--num_rank", type=int, default=3, help="tensor are decomposed with this rank")
    parser.add_argument("--equal_vdn", action='store_true',
                        default=False, help="Whether to make the algorithm equal to vdn")
    parser.add_argument("--msg_anytime", action='store_false',
                        default=True, help="Anytime extension of greedy action selection (Kok and Vlassis, 2006)")
    parser.add_argument("--msg_normalized", action='store_false',
                        default=True, help="Message normalization during greedy action selection (Kok and Vlassis, 2006)")
    parser.add_argument("--lamda", type=float, default=0, 
                        help="Damping factor for messaging")
    parser.add_argument("--msg_iterations", type=int, default=4,
                        help="Number of cycles of factor graph message passing algorithm")
    parser.add_argument('--highest_orders', type=int, default=3, help="number of agents")
    #     adj network parameters
    parser.add_argument('--adj_hidden_dim', type=int, default=32,
                        help="Dimension of hidden layers for adj networks")
    parser.add_argument('--adj_output_dim', type=int, default=32,
                        help="Dimension of output layers for adj networks")
    parser.add_argument('--adj_alpha', type=float, default=0.1,
                        help="alpha")
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help="entropy term coefficient (default: 0.2)")
    parser.add_argument("--use_linear_lr_decay", action='store_false',
                        default=True, help='use a linear schedule on the learning rate')
    parser.add_argument("--entropy_coef", type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--use_valuenorm", action='store_true', default=False, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_vfunction", action='store_false', default=True)
    parser.add_argument("--use_epsilon_greedy", action='store_true', default=False)
    parser.add_argument("--pretrain_adj", action='store_true', default=False)
    parser.add_argument("--adj_max_grad_norm", type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument('--num_mini_batch', type=int, default=1)
    parser.add_argument('--adj_begin_step', type=int, default=0)
    parser.add_argument("--use_adj_init", action='store_false', default=True)
    parser.add_argument('--num_factor', type=int, default=28, help="number of factor")
    
    # exploration parameters
    parser.add_argument('--num_random_episodes', type=int, default=5,
                        help="Number of episodes to add to buffer with purely random actions")
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help="Starting value for epsilon, for eps-greedy exploration")
    parser.add_argument('--epsilon_finish', type=float, default=0.05,
                        help="Ending value for epsilon, for eps-greedy exploration")
    parser.add_argument('--epsilon_anneal_time', type=int, default=50000,
                        help="Number of episodes until epsilon reaches epsilon_finish")
    parser.add_argument('--adj_anneal_time', type=int, default=800000,
                        help="Number of episodes until epsilon reaches epsilon_finish")
    parser.add_argument('--disount_step', type=int, default=500000,
                        help="Discount factor of computational adj networks")
    parser.add_argument('--act_noise_std', type=float,
                        default=0.1, help="Action noise")

    # train parameters
    parser.add_argument('--actor_train_interval_step', type=int, default=2,
                        help="After how many critic updates actor should be updated")
    parser.add_argument('--train_interval_episode', type=int, default=1,
                        help="Number of env steps between updates to actor/critic")
    parser.add_argument('--train_adj_episode', type=int, default=4,
                        help="Number of env steps between train adj network")
    parser.add_argument('--drop_temperature_episode', type=int, default=10,
                        help="Number of env steps between drop_temperature")
    parser.add_argument('--train_interval', type=int, default=100,
                        help="Number of episodes between updates to actor/critic")
    parser.add_argument("--use_value_active_masks",
                        action='store_true', default=False)
    parser.add_argument('--use_available_actions', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")

    # eval parameters
    parser.add_argument('--use_eval', action='store_false',
                        default=True, help="Whether to conduct the evaluation")
    parser.add_argument('--eval_interval', type=int,  default=10000,
                        help="After how many episodes the policy should be evaled")
    parser.add_argument('--num_eval_episodes', type=int, default=32,
                        help="How many episodes to collect for each eval")

    # save parameters
    parser.add_argument('--save_interval', type=int, default=100000,
                        help="After how many episodes of training the policy model should be saved")
    parser.add_argument('--use_save', action='store_false',
                        default=True, help="Whether to save the model")

    # log parameters
    parser.add_argument('--log_interval', type=int, default=1000,
                        help="After how many episodes of training the policy model should be saved")

    # pretained parameters
    parser.add_argument("--model_dir", type=str, default=None)

    #webots experiment parameters(pp)
    parser.add_argument("--num_obs_targets", type=int, default=2)
    parser.add_argument("--num_obs_agents", type=int, default=2)
    parser.add_argument("--radius_view", type=float, default=0.4)
    parser.add_argument("--catch_distance", type=float, default=0.18)
    parser.add_argument("--collision_distance", type=float, default=0.04)
    parser.add_argument("--collision_reward", type=float, default=-0.1)
    parser.add_argument("--timestep", type=int, default=100)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--num_stags", type=int, default=4)
    parser.add_argument('--num_agents', type=int,default=6, help="number of agents")
    parser.add_argument('--capture_action_conditions', type=list, default=[2, 1],help="number of agents that have to simultaneously execute catch action")
    parser.add_argument('--reward_stag', type=int, default=10, help="reward for capturing a stag")
    parser.add_argument('--reward_time', type=float,default=0, help="reward (or punishment) given at each time step")
    parser.add_argument('--miscapture_punishment', type=float, default=-2)
    parser.add_argument('--capture_action', action='store_false',default=True,
                        help="whether capturing requires an extra action (True) or just capture_conditions (False)")


    
    return parser
