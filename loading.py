import argparse
from itertools import count
import numpy as np
import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--phi', type=float, default=1e-1, metavar='G',
                    help='ARPL perturbation prob (default: 1e-1)')
parser.add_argument('--eps', type=float, default=1, metavar='G',
                    help='ARPL perturbation strength (default: 1e-2')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render',action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

policy_net.load_state_dict(torch.load('./weights/weights_gan/Weights_policy_proReacher-v2.pt'))

phi = args.phi
epsilon = args.eps
phi = 0.0
epsilon = 0.0
tol = 2.0
early = 0

running_state = ZFilter((num_inputs,), clip=5)


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def run_episode(phi):
    reward_sum = 0
    state = env.reset()
    state = running_state(state)
    state_dist = []
    while True:
        action = select_action(state)
        action = action.data[0].numpy()
        next_state, reward, done, _ = env.step(action)
        flag = np.float(np.random.choice([0,1],p=[1-phi,phi])) # to perturb with probability phi
        if flag:
            next_state = next_state + [0,0,0,0,0.005*np.random.randn(),0.005*np.random.randn(),0.005*np.random.randn(),0.005*np.random.randn(),0.005*np.random.randn(),0.005*np.random.randn(),0]
        reward_sum += reward
        state_dist.append(state[-3:-1])
        # print("1 "+str(next_state))
        next_state = running_state(next_state)
        # print("2 "+str(next_state))
        #if args.render:
        # env.render()
        if done:
            break
        state = next_state
    # print(np.min(np.abs(np.array(state_dist)),axis=0))
    return reward_sum


reward_plot = []
std_plot = []
for phi in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    print("Starting Evaluation for 1000 episodes "+"Phi= "+str(phi))
    rew_arr = []
    count = 0
    t_rew = 0
    for i in range(1000):
        rew = run_episode(phi)
        t_rew += rew
        rew_arr.append(rew)
        count += 1
    avg_rew = t_rew/count
    std_rew = np.std(np.array(rew_arr))
    print("Average Reward: "+str(avg_rew))
    print("Standard Deviation: "+str(std_rew))
    reward_plot.append(avg_rew)
    std_plot.append(std_rew)
np.save(file='rew_arr_gan_0_01',arr=np.array(reward_plot))
np.save(file='std_arr_gan_0_01.npy',arr=np.array(std_plot))