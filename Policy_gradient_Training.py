import numpy as np
from agent import env
from policy_gradient import *
agent = Agent()


Myenv = env(15)
score_history = []

num_episodes = 2000

for i in range(num_episodes):
    done = False
    score = 0
    Myenv = env(15)
    observation = np.array(Myenv.getFeature()[0], dtype = np.float32)
    while not done:
        action = agent.choose_action(observation)
        Myenv.step(action)
        observation_, done, reward = Myenv.getFeature()
        agent.store_transition(observation, action, reward)
        observation = np.array(observation_, dtype = np.float32)
        score += reward
    score_history.append(score)

    agent.learn()
    avg_score = np.mean(score_history[-100:])
    print('episode: ', i,'score: %.1f' % score,
        'average score %.1f' % avg_score)
agent.model_save('model/PG_MODEL.pth')