from dqnn import Agent
import numpy as np
import gym
from utils import plotLearning

import tensorflow as tf

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    env = gym.make('LunarLander-v2')
    lr = 0.001
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1.0,
                  input_dims=env.observation_space.shape[0], lr=lr, n_games=n_games, mem_size=100000, batch_size=64, epsilon_dec=0.91)
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action,
                                   reward, observation_, done)
            observation = observation_
            score += reward
            agent.learn()
        scores.append(score)
        eps_history.append(agent.epsilon)
        agent.learn()

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score, 'avg score: ',
              avg_score, 'epsilon: ', agent.epsilon)
        
        filename = 'dqn_lunar_v2_' + str(i) + '.png'
        x = [i+1 for i in range(len(scores))]
        plotLearning(x, scores, eps_history, filename)
