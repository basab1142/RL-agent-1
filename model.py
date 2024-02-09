# imports
from agent import *
import numpy as np
import torch
import random
import torch.nn.functional as F

# buffer space
from collections import deque
replay_buffer = deque(maxlen = 100_00) # Buffer space
NUM_INPUTS = 8
HIDDEN_UNITS = 20
OUTPUTS = 4
BATCH_SIZE = 64
GAMMA = 0.95

# Building model
import torch.nn as nn

class CaveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputLayer = nn.Linear(in_features=NUM_INPUTS, out_features=HIDDEN_UNITS)
        self.hiddenLayer = nn.Linear(in_features=HIDDEN_UNITS, out_features=HIDDEN_UNITS)
        self.outputLayer = nn.Linear(in_features=HIDDEN_UNITS, out_features= OUTPUTS)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.inputLayer(x)
        x = self.relu(x)
        x = self.hiddenLayer(x)
        x = self.relu(x)
        x = self.hiddenLayer(x)
        x = self.relu(x)
        out = self.outputLayer(x)
        return out
    
# initiate poilicy and target model
model = CaveNet()
target_model = CaveNet()
# Action selection function
@torch.no_grad()
def epsilon_greedy_action_selection(model, epsilon, observation, sharpening_factor = 1):
    if np.random.random() > epsilon:
        prediction = model(torch.tensor(observation, dtype = torch.float))  # perform the prediction on the observation
        # Chose the action from softmax distribution
        action = torch.multinomial(F.softmax(prediction*sharpening_factor, dim = 0), num_samples = 1).item()   
    else:
        action = np.random.randint(0, 4)  # Else use random action
    return action        

# set up optimizer and loss function
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.1)
loss_fn = torch.nn.L1Loss()

# Training Loop
def train(states, target, model = model, EPOCHS = 1, batch_size = BATCH_SIZE):
    for epoch in range(EPOCHS):
        # train mode on
        model.train()
        # forward prop
        y_preds = model(states)
        loss = loss_fn(y_preds, target)
        #optimizer zero grading
        optimizer.zero_grad()
        # backprop the loss
        loss.backward()
        #optimizer step
        optimizer.step()

# Replay memory implementaion
def replay(replay_buffer,batch_size = BATCH_SIZE, model = model, target_model = target_model ):
    
    if len(replay_buffer) < batch_size:
        return
    samples = random.sample(replay_buffer, batch_size)
    target_batch = []  
    
    zipped_samples = list(zip(*samples))  
    states, actions, rewards, new_states, dones = zipped_samples
    states, new_states = torch.tensor(np.array(states), dtype = torch.float), torch.tensor(np.array(new_states), dtype = torch.float)
    with torch.inference_mode():
        # Predict targets for all states from the sample
        targets = target_model(states)
        # Predict Q-Values for all new states from the sample
        q_values = model(new_states)
    for i in range(batch_size):
        # take the maximum value
        q_val = max(q_values[i])
        target = torch.clone(targets[i]).numpy()
        if dones[i]:
            target[actions[i]] = rewards[i]
        else:
            target[actions[i]] = rewards[i] + q_val * GAMMA
        
        target_batch.append(target)
    train(states, torch.tensor(target_batch))    


# EPSIOLN and it's reduction parameters
EPSILON = 1.0
EPSILON_REDUCE = 0.9


# Actual training
def training(model = model, target_model = target_model,EPSILON = EPSILON, EPSILON_REDUCE = EPSILON_REDUCE,EPOCHS = 1):

    
    num_done = 0
    for epoch in range(EPOCHS):
        MyEnv = env(15) # initialsation of environment
        state, done, reward = MyEnv.getFeature()
        num_simulation = 0 # we will simulate an episode for maximum of 100 steps
        while not done:
            # choose an action
            action = epsilon_greedy_action_selection(model, EPSILON, state)
            # perform action and get next state
            MyEnv.step(action)
            new_state, done, reward = MyEnv.getFeature()
            replay_buffer.append((state, action, reward, new_state, done))
            state = new_state
            if reward == 10:
                num_done += 1
            num_simulation += 1
            if num_simulation >= 100:
                break
            
        replay(replay_buffer)  
       
        EPSILON *= EPSILON_REDUCE
        
        if epoch % 500 == 0:
            target_model.load_state_dict(model.state_dict())
            print(f" {epoch} : DONES = {num_done}")    

# Train the model [Uncomment to train]
#training(model, target_model,EPSILON,EPSILON_REDUCE,EPOCHS = 600)                        