
import collections, random, itertools
import gym
from  lib.envs.windyGridworld import WindyGridworldEnv
from lib.envs.cliffwalking import CliffWalkingEnv
import numpy as np
from lib import plots


class nStepSARSA(object):

    def __init__(self, env, num_episode, n, discount = 1.0, alpha=0.2, epsilon = .1) -> None:
        self.env = env
        self.num_episode = num_episode
        self.n = n
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon

    def epsilonGreedyPolicy(self, Q):
        def policy(state):
            act = self.epsilon * np.ones(len(Q[state]), dtype=float)/len(Q[state])
            act[np.argmax(Q[state])] += 1 - self.epsilon
            return act
        return policy

    def nStepSARSA(self):
        Q = collections.defaultdict(lambda:np.zeros(self.env.action_space.n, dtype=float))
        policy = self.epsilonGreedyPolicy(Q)

        statsVal = plots.EpisodeStats(
        episode_lengths=np.zeros(self.num_episode),
        episode_rewards=np.zeros(self.num_episode))
        

        for episode in range(self.num_episode):
            G, states, actions, rewards, T = 0, [], [], [], float('inf')
            state = self.env.reset()
            states.append(state)
            action = np.random.choice(np.arange(self.env.action_space.n), p = policy(state))
            actions.append(action)

            t = -1
            while (t<T-1):
                t += 1
                if t < T:
                    nextState, reward, done, info = self.env.step(action)
                    rewards.append(reward)
                    states.append(nextState)

                    if done:
                        T = t+1
                    else:
                        next_action = np.random.choice(np.arange(self.env.action_space.n), p=policy(nextState))
                        actions.append(next_action)
                    statsVal.episode_rewards[episode] += reward
                    statsVal.episode_lengths[episode] = t
                tauValue = t - self.n +1
                if tauValue >= 0:

                    def calculate_G():
                        for i in range(tauValue, min(tauValue+self.n, T)):
                            for j in range(tauValue, min(tauValue+self.n, T)):
                                G += self.discount**(j-tauValue)*rewards[j]
                        if tauValue + self.n < T:
                            G += self.discount**self.n*Q[states[tauValue+self.n]][actions[tauValue+self.n]]
                            
                        def update_Q():
                            Q[states[tauValue]][actions[tauValue]] += self.alpha*(G-Q[states[tauValue]][actions[tauValue]])
                            return
                        update_Q()

                    calculate_G()
                    
                if tauValue == T-1:
                    break
        return Q,stats, tauValue, G, policy

            



if __name__ == "__main__":
    env1 = WindyGridworldEnv()
    env2 = CliffWalkingEnv()
    env3 = gym.make("CartPole-v01")
    nss = nStepSARSA(env1, 20,5)
    q,stats, tauValue, G, policy = nss.nStepSARSA()
    plots.plot_episode_stats(stats)
    
