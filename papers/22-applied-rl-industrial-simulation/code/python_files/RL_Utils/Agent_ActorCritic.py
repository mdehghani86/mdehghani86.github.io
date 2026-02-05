import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Force Use a Device
# device = 'cuda' #for GPU
# device = 'cpu'  #for CPU

class ActorNeuralNetwork(nn.Module):

    def __init__(self, input_dims, num_actions):

        super(ActorNeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(*input_dims, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)

        self.mu = nn.Linear(16,num_actions)
        self.sigma = nn.Linear(16,num_actions)

    def forward(self, x):

        l1 = self.layer1(x)
        l1 = F.relu(l1)
        l2 = self.layer2(l1)
        l2 = F.relu(l2)
        l3 = self.layer3(l2)

        mu = self.mu(l3)

        sigma = self.sigma(l3)
        sigma = torch.sigmoid(sigma)

        return mu, sigma

class CriticNeuralNetwork(nn.Module):

    def __init__(self, input_dims, num_actions):

        super(CriticNeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(*input_dims, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, num_actions)

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
                 alr=1e-4, blr=1e-4, gamma=0.9, epsilon_decay=(1- (1e-4)), epsilon_min=0):

        self.input_dims = input_dims
        self.gamma = gamma

        self.actor_network = ActorNeuralNetwork(
            input_dims=input_dims, num_actions=num_actions).to(device)

        self.critic_network = CriticNeuralNetwork(
            input_dims=input_dims, num_actions=num_actions).to(device)

        self.optimizer_actor = torch.optim.Adam(
            self.actor_network.parameters(), lr=alr)

        self.optimizer_critic = torch.optim.Adam(
            self.critic_network.parameters(), lr=blr)
        
        self.log_probs = None

        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, obs):

        obs_T = torch.tensor([obs], device=device).float()

        mu, sigma = self.actor_network(obs_T)

        action_probs = Normal(mu, sigma)

        if(np.random.random() < self.epsilon):
            probs = action_probs.rsample()
        else:
            probs = action_probs.sample() 

        self.log_probs = action_probs.log_prob(probs).to(device)

        action = torch.sigmoid(probs)

        return action.cpu().detach().numpy()[0]

    def train(self, obs, new_obs, reward, done):

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        critic_value_ = self.critic_network(torch.tensor(new_obs).float().to(device))
        critic_value = self.critic_network(torch.tensor(obs).float().to(device))

        reward = torch.tensor(reward).float().to(device)

        delta = reward + (self.gamma*critic_value_*(1-int(done))) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        self.epsilon = max((self.epsilon*self.epsilon_decay), self.epsilon_min)

        return actor_loss.item(), critic_loss.item()

    def save_model(self, file_path='./rl_models/ActorCritic/'):
        torch.save(self.actor_network.state_dict(), file_path+"actor_network.model")
        torch.save(self.critic_network.state_dict(), file_path+"critic_network.model")

    def load_model(self, file_path='./rl_models/ActorCritic/'):
        self.actor_network.load_state_dict(torch.load(file_path+"actor_network.model"))
        self.critic_network.load_state_dict(torch.load(file_path+"critic_network.model"))