from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
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


class NeuralNetwork(nn.Module):

    def __init__(self, input_dims, n_actions):

        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(*input_dims, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, n_actions)

    def forward(self, x):

        l1 = self.layer1(x)
        l1 = F.relu(l1)
        l2 = self.layer2(l1)
        l2 = F.relu(l2)
        l3 = self.layer3(l2)

        output = l3

        return output


class Agent():
    def __init__(self, n_actions, input_dims,
                 lr=1e-4, gamma=0.9, mem_size=128, batch_size=64,
                 epsilon_decay=0.995, epsilon_min=0, target_update_frequency=256):

        self.n_actions = n_actions
        self.input_dims = input_dims

        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.target_update_freq = target_update_frequency

        self.policy_network = NeuralNetwork(
            input_dims=input_dims, n_actions=n_actions).to(device)
        self.target_network = deepcopy(self.policy_network)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=lr)

        self.replay_mem = ReplayBuffer(
            mem_size=mem_size, batch_size=batch_size, input_dims=input_dims)

        self.epsilon = 1

    def choose_action(self, obs):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            obs_T = torch.tensor(obs, device=device).float()
            with torch.no_grad():
                policy_values = self.policy_network(
                    obs_T).cpu().detach().numpy()
            action = np.argmax(policy_values)

        return (action+1)

    def store_memory(self, state, action, reward, new_state, done):
        self.replay_mem.store_transitions(
            state, (action-1), reward, new_state, done)

    def train(self):

        if not(self.replay_mem.is_sampleable()):
            return 0

        states, new_states, actions, rewards, dones = self.replay_mem.sample_buffer()

        states_T = torch.tensor(states, device=device).float()
        new_states_T = torch.tensor(new_states, device=device).float()
        rewards_T = torch.tensor(rewards, device=device).float()
        dones_T = torch.tensor(dones, device=device)
        actions_T = torch.tensor(actions, device=device).type(
            torch.int64).unsqueeze(1)

        q_eval = self.policy_network(states_T).gather(1, actions_T).squeeze(1)

        q_next = self.target_network(new_states_T).max(1)[0].detach()
        q_next[dones_T] = 0.0

        q_target = rewards_T + (self.gamma * q_next)

        loss = self.loss_function(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max((self.epsilon * self.epsilon_decay), self.epsilon_min)

        if(self.replay_mem.mem_centr % self.target_update_freq == 0):
            self.target_network = deepcopy(self.policy_network)

        return loss.item()

    def save_model(self, file_path='./rl_models/DDQN/'):
        torch.save(self.policy_network.state_dict(), file_path+"policy_ddqn_model.model")
        torch.save(self.target_network.state_dict(), file_path+"target_ddqn_model.model")

    def load_model(self, file_path='./rl_models/DDQN/'):
        self.policy_network.load_state_dict(torch.load(file_path+"policy_ddqn_model.model"))
        self.target_network.load_state_dict(torch.load(file_path+"target_ddqn_model.model"))
