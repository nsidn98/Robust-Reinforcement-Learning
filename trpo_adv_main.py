import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
import matplotlib.style
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('classic')

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--phi', type=float, default=1e-1, metavar='G',
                    help='ARPL perturbation prob (default: 1e-1)')
parser.add_argument('--eps', type=float, default=0, metavar='G',
                    help='ARPL perturbation strength (default: 1e-2')
parser.add_argument('--curriculum', type=int, default=50, metavar='G',
                    help='ARPL perturbation curriculum (default: 100')
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
# print(str(args.eps)+args.env_name)
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net_pro = Policy(num_inputs, num_actions)
value_net_pro = Value(num_inputs)
policy_net_pro.load_state_dict(torch.load("Weights_policy_proReacher-v2.pt"))
value_net_pro.load_state_dict(torch.load("Weights_value_proReacher-v2.pt"))

policy_net_adv = Policy(num_inputs, num_actions)
value_net_adv = Value(num_inputs)
# policy_net_adv.load_state_dict(torch.load("Weights_policy_advReacher-v2.pt"))
# value_net_adv.load_state_dict(torch.load("Weights_value_advReacher-v2.pt"))

phi = args.phi
epsilon = args.eps
# phi = 0.2
# epsilon = 0.1
tol = 2.0
early = 0
gan_interval = 120

def select_action(policy_net,value_net,state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(policy_net,value_net,batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))
    # print(states)

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)


plt.ion()
reward_plot = []
reward_batch_0 = 1000
for i_episode in count(1):
    memory_pro = Memory()
    memory_adv = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action_pro = select_action(policy_net_pro,value_net_pro,state)
            action_pro = action_pro.data[0].numpy()
            action_adv = select_action(policy_net_adv,value_net_adv,state)
            action_adv = action_adv.data[0].numpy()
            action_sum = action_pro + 0.01*action_adv  # can take mean/random choice also
            next_state, reward, done, _ = env.step(action_sum)
            reward = np.max([-1500.0,reward]) # clip rewards to -1500
            reward_sum += reward
            next_state = running_state(next_state)

            mask = 1

            if done:
                mask = 0

            memory_pro.push(state, np.array([action_pro]), mask, next_state, reward)
            memory_adv.push(state, np.array([action_adv]), mask, next_state, -1000.0*reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    train_interval = i_episode%gan_interval
    # if train_interval < 100:
    batch_pro = memory_pro.sample()
    print("Protogonist Training")
    update_params(policy_net_pro,value_net_pro,batch_pro)
    
    # else:
    batch_adv = memory_adv.sample()
    # print("Adversary Training")
    update_params(policy_net_adv,value_net_adv,batch_adv)
    reward_plot.append(reward_batch)
    plt.title(str(args.env_name)+r' Training Rewards Plot for $\varepsilon$ = '+str(epsilon))
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.plot(np.arange(i_episode),reward_plot,linestyle='--', marker='D', color='k')
    plt.pause(1e-3)
    plt.clf()
    percentage_change = 100*np.abs((reward_batch - reward_batch_0)/reward_batch_0)
    reward_batch_0 = reward_batch
    # print(reward_batch_0)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}\tPercentage Change {}'.format(
            i_episode, reward_sum, reward_batch,percentage_change))
    if i_episode%args.curriculum == 0:
        print("Changing" +r"$\Phi$")
        phi = phi+0.005

    if percentage_change < tol:
        early += 1
    else:
        early = 0
    
    if early >= 5:
        print("Saving Weights")
        torch.save(policy_net_pro.state_dict(),"Weights_policy_pro"+args.env_name+".pt")
        torch.save(value_net_pro.state_dict(),"Weights_value_pro"+args.env_name+".pt")        
        torch.save(policy_net_adv.state_dict(),"Weights_policy_adv"+args.env_name+".pt")
        torch.save(value_net_adv.state_dict(),"Weights_value_adv"+args.env_name+".pt")        
        np.save(file="Plot_array_adv_trpo.npy",arr=np.array(reward_plot))

        continue

    if i_episode%100 == 0:
        print("Saving Weights")
        torch.save(policy_net_pro.state_dict(),"Weights_policy_pro"+args.env_name+".pt")
        torch.save(value_net_pro.state_dict(),"Weights_value_pro"+args.env_name+".pt")        
        torch.save(policy_net_adv.state_dict(),"Weights_policy_adv"+args.env_name+".pt")
        torch.save(value_net_adv.state_dict(),"Weights_value_adv"+args.env_name+".pt")        
        np.save(file="Plot_array_adv_trpo.npy",arr=np.array(reward_plot))
        continue

plt.show()