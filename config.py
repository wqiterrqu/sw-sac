import argparse

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval_interval', type=int, default="10",metavar='N',
                    help='Eval interval (default: 10)')
parser.add_argument('--eval_episodes', type=int, default="10",metavar='N',
                    help='Eval episodes (default: 10)')
parser.add_argument('--save_checkpoint', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--theta', type=float, default=1, metavar='G',
                    help='theta value (default: 0.99)')
parser.add_argument('--reward_factor', type=float, default=1.001, metavar='G',
                    help='')
parser.add_argument('--reward_multiple', type=float, default=1.0, metavar='G',
                    help='')
parser.add_argument('--tolerator', type=float, default=0.80, metavar='G',
                    help='')
parser.add_argument('--best_reward_interval', type=int, default=10, metavar='N',
                    help='')
parser.add_argument('--max_reward_multiple', type=float, default=1.5, metavar='G',
                    help='')
parser.add_argument('--rsample', type=bool, default=True, metavar='G',
                    help='if use rsample')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type =int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--env_name', default="HalfCheetah-v4",
                    help='Mujoco Gym environment (default: HalfCheetah-v4)')
parser.add_argument('--algorithm_name', default="sac",
                    help='algorithm')
parser.add_argument('--experiment_name', default="RewardTrick",
                    help='-')
parser.add_argument('--beta', type=float, default=0.3, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--RewardType', default="sliding_window",
                    help='baseline or sliding_window')
parser.add_argument('--SlideLength', type=int, default=2, metavar='N',
                    help='length of the sliding window')
parser.add_argument('--info', default="")
parser.add_argument('--info', default="")

args = parser.parse_args()
