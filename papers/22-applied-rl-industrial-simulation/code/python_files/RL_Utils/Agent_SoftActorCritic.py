import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Force Use a Device
# device = 'cuda' #for GPU
# device = 'cpu'  #for CPU

class ReplayBuffer():
    def __init__(self, mem_size, batch_size, input_dims):
        self.mem_size = mem_size
        self.mem_centr = 0
        self.batch_size = batch_size

        self.state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transitions(self, state, action, reward, new_state, done):
        index = self.mem_centr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_centr = self.mem_centr + 1

    def is_sampleable(self):
        if self.mem_centr >= self.batch_size:
            return True
        else:
            return False

    def sample_buffer(self):
        if not(self.is_sampleable()):
            return []

        max_mem = min(self.mem_size, self.mem_centr)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, new_states, actions, rewards, terminals


class ActorNeuralNetwork(nn.Module):

    def __init__(self, input_dims, num_actions):

        super(ActorNeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(*input_dims, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 128)

        self.mu = nn.Linear(128,num_actions)
        self.sigma = nn.Linear(128,num_actions)

        self.reparam_noise = 1e-6

    def forward(self, x):

        l1 = self.layer1(x)
        l1 = F.relu(l1)
        l2 = self.layer2(l1)
        l2 = F.relu(l2)
        l3 = self.layer3(l2)

        mu = self.mu(l3)
        mu = torch.tanh(mu)*5.0

        sigma = self.sigma(l3)
        sigma = torch.sigmoid(sigma)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = 0.5 * (torch.tanh(actions) + 1)

        log_probs = probabilities.log_prob(actions)

        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)

        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

class CriticNeuralNetwork(nn.Module):

    def __init__(self, input_dims, num_actions):

        super(CriticNeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(input_dims[0] + num_actions, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=1)

        l1 = self.layer1(x)
        l1 = F.relu(l1)
        l2 = self.layer2(l1)
        l2 = F.relu(l2)
        l3 = self.layer3(l2)

        output = l3

        return output

class ValueNeuralNetwork(nn.Module):

    def __init__(self, input_dims):

        super(ValueNeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(*input_dims, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):

        l1 = self.layer1(x)
        l1 = F.relu(l1)
        l2 = self.layer2(l1)
        l2 = F.relu(l2)
        l3 = self.layer3(l2)

        output = l3

        return output

class Agent():
    def __init__(self, input_dims, num_actions=1,
                 alr=1e-4, blr=1e-4, tau=0.005, gamma=0.9, mem_size=128, batch_size=64,
                 reward_scale=1):

        self.input_dims = input_dims

        self.tau = tau
        self.gamma = gamma

        self.batch_size = batch_size
        self.scale = reward_scale

        self.actor_network = ActorNeuralNetwork(
            input_dims=input_dims, num_actions=num_actions).to(device)

        self.critic1_network = CriticNeuralNetwork(
            input_dims=input_dims, num_actions=num_actions).to(device)
        self.critic2_network = CriticNeuralNetwork(
            input_dims=input_dims, num_actions=num_actions).to(device)
        
        self.value_network = ValueNeuralNetwork(
            input_dims=input_dims).to(device)
        self.targetvalue_network = ValueNeuralNetwork(
            input_dims=input_dims).to(device)

        self.optimizer_actor = torch.optim.Adam(
            self.actor_network.parameters(), lr=alr)

        self.optimizer_critic1 = torch.optim.Adam(
            self.critic1_network.parameters(), lr=blr)
        self.optimizer_critic2 = torch.optim.Adam(
            self.critic2_network.parameters(), lr=blr)

        self.optimizer_value = torch.optim.Adam(
            self.value_network.parameters(), lr=blr)
        self.optimizer_targetvalue = torch.optim.Adam(
            self.targetvalue_network.parameters(), lr=blr)

        self.replay_mem = ReplayBuffer(
            mem_size=mem_size, batch_size=batch_size, input_dims=input_dims)
        
        self.update_network_parameters(tau=1)

    def choose_action(self, obs):
        obs_T = torch.tensor([obs], device=device).float()
        action, _ = self.actor_network.sample_normal(obs_T, reparameterize=False)

        return action.cpu().detach().numpy()[0]

    def store_memory(self, state, action, reward, new_state, done):
        self.replay_mem.store_transitions(
            state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        target_value_params = self.targetvalue_network.named_parameters()
        value_params = self.value_network.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)* target_value_state_dict[name].clone()

        self.targetvalue_network.load_state_dict(value_state_dict)

    def train(self):

        if not(self.replay_mem.is_sampleable()):
            return 0.0, 0.0, 0.0

        states, new_states, actions, rewards, dones = self.replay_mem.sample_buffer()

        states_T = torch.tensor(states, device=device).float()
        new_states_T = torch.tensor(new_states, device=device).float()
        rewards_T = torch.tensor(rewards, device=device).float()
        dones_T = torch.tensor(dones, device=device)
        actions_T = torch.tensor(actions, device=device).float().unsqueeze(1)

        value = self.value_network(states_T).squeeze(1)
        value_ = self.targetvalue_network(new_states_T).squeeze(1)
        value_[dones_T] = 0.0

        training_actions, log_probs = self.actor_network.sample_normal(states_T, reparameterize=False)
        log_probs = log_probs.squeeze(1)

        q1_new_policy = self.critic1_network(states_T, training_actions)
        q2_new_policy = self.critic2_network(states_T, training_actions)

        critic_value = torch.min(q1_new_policy, q2_new_policy).squeeze(1)

        self.optimizer_value.zero_grad()
        target_value = critic_value - log_probs
        value_loss = F.mse_loss(value, target_value)
        value_loss.backward(retain_graph=True)
        self.optimizer_value.step()

        training_actions, log_probs = self.actor_network.sample_normal(states_T, reparameterize=True)
        log_probs = log_probs.squeeze(1)
        q1_new_policy = self.critic1_network(states_T, training_actions)
        q2_new_policy = self.critic2_network(states_T, training_actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy).squeeze(1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.optimizer_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizer_actor.step()

        q_hat = (self.scale*rewards_T) + (self.gamma * value_)

        q1_old_policy = self.critic1_network(states_T, actions_T).squeeze(1)
        q2_old_policy = self.critic2_network(states_T, actions_T).squeeze(1)

        
        self.optimizer_critic1.zero_grad()
        self.optimizer_critic2.zero_grad()

        critic1_loss = F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic1_loss + critic2_loss

        critic_loss.backward()

        self.optimizer_critic1.step()
        self.optimizer_critic2.step()

        self.update_network_parameters()

        return actor_loss.item(), critic_loss.item(), value_loss.item()

    def save_model(self, file_path='./rl_models/SAC/'):
        torch.save(self.actor_network.state_dict(), file_path+"actor_network.model")
        torch.save(self.critic1_network.state_dict(), file_path+"critic1_network.model")
        torch.save(self.critic2_network.state_dict(), file_path+"critic2_network.model")
        torch.save(self.value_network.state_dict(), file_path+"value_network.model")
        torch.save(self.targetvalue_network.state_dict(), file_path+"targetvalue_network.model")

    def load_model(self, file_path='./rl_models/SAC/'):
        self.actor_network.load_state_dict(torch.load(file_path+"actor_network.model"))
        self.critic1_network.load_state_dict(torch.load(file_path+"critic1_network.model"))
        self.critic2_network.load_state_dict(torch.load(file_path+"critic2_network.model"))
        self.value_network.load_state_dict(torch.load(file_path+"value_network.model"))
        self.targetvalue_network.load_state_dict(torch.load(file_path+"targetvalue_network.model"))