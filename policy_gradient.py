import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch
import numpy as np

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features= 8,out_features= 256)
        self.layer2 = nn.Linear(in_features= 256, out_features = 4)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, x):
        return self.softmax(self.layer2(self.ReLU(self.layer1(x))))
    

class Agent:
    def __init__(self, alpha=0.003, gamma=0.99):

        self.gamma = gamma
        self.lr = alpha
        self.n_actions = 4
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy = Policy()
        self.optimizer = optim.AdamW(params=self.policy.parameters(), lr = alpha)
        

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float32)
        probs = self.policy(state)
        action_probs = Categorical(probs=probs)
        action = action_probs.sample().item()

        return action

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
    
    def model_save(self, PATH = 'PG_MODEL.pth'):
        
        torch.save(self.policy.state_dict(), PATH)

    def learn(self):
        actions = torch.tensor(self.action_memory, dtype=torch.float32)
        rewards = np.array(self.reward_memory)

        G = torch.zeros_like(torch.tensor(rewards))
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        
        self.policy.train()
        loss = 0
        
        for idx, (g, state) in enumerate(zip(G, self.state_memory)):
            state = torch.tensor(state, dtype= torch.float32)
            probs = self.policy(state)
            action_probs = Categorical(probs=probs)
            log_prob = action_probs.log_prob(actions[idx])
            loss += -g * log_prob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []    


def action_selection(policy, observation):
        state = torch.tensor(observation, dtype=torch.float32)
        probs = policy(state)
        action_probs = Categorical(probs=probs)
        action = action_probs.sample().item()

        return action        