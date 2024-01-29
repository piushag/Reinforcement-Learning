from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import numpy as np
import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

class TempControlEnv(Env):
  def __init__(self):
    self.action_space = Discrete(3)
    self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([100, 120, 1]) )
    self.length = 60000
    self.temp_change_rate = random.uniform(- 0.05, 0.05)
    self.room_temp = 72 + random.uniform(-5,5)
    self.outside_temp = 72 + random.uniform(-25,25)
    self.state = [self.room_temp, self.outside_temp, self.temp_change_rate]
    self.k = .1


  def step(self, action):
    self.outside_temp += self.temp_change_rate
    self.state[0] += (action + self.k*(self.outside_temp - self.room_temp) - 1)
    self.length -= 1

    if self.state[0] >= 65 and self.state[0] <=85:
      reward = 0
    else:
      reward =-1

    if self.length <= 0:
      done = True
    else:
      done = False

    info = {}

    return self.state, reward, done, info

  def render(self):
    pass

  def reset(self):
    self.length = 60000
    self.temp_change_rate = random.uniform(- 0.05, 0.05)
    self.room_temp = 72 + random.uniform(-5,5)
    self.outside_temp = 72 + random.uniform(-25,25)
    self.state = [self.room_temp, self.outside_temp, self.temp_change_rate]
    return self.state

class test:
  def __init__(self):
    self.env = TempControlEnv()
    env.observation_space.sample()

  def generate_random_episode(self):
      episodes = 10
      ans = []
      for episode in range(0, episodes):
          state = env.reset()
          done = False
          score = 0 
          
          while not done:
              #env.render()
              action = env.action_space.sample()
              n_state, reward, done, info = env.step(action)
              score+=reward
          ans.append(score)

      print(np.mean(ans))

  
  def build_model(self,states, actions):
      model =  tensorflow.keras.Sequential()
      model.add(Flatten(input_shape=(1,states)))
      model.add(Dense(48, activation='relu'))
      model.add(Dense(48, activation='relu'))
      model.add(Dense(48, activation='relu'))
      model.add(Dense(48, activation='relu'))
      model.add(Dense(48, activation='relu'))
      model.add(Dense(actions, activation='linear'))
      return model

  def build_agent(self, model, actions):
      policy = BoltzmannQPolicy()
      memory = SequentialMemory(limit=50000, window_length=1)
      dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                    nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
      return dqn

if __name__ == "__main__":
    t = test()
    states = t.env.observation_space.shape[0]
    print(states) 
    actions = t.env.action_space.n

    from stable_baselines3.common.evaluation import evaluate_policy
    evaluate_policy(model, env, n_eval_episodes=10, render=False)

    from stable_baselines3 import DQN

    model = DQN('MlpPolicy', env, verbose = 1)

    model.learn(total_timesteps=60000)
    evaluate_policy(model, env, n_eval_episodes=10, render=False)
    
    model = t.build_model(states, actions)
    model.summary()

    dqn = t.build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(t.env, nb_steps=20000, visualize=False, verbose=1)

    scores = dqn.test(t.env, nb_episodes=10, visualize=False)
    print(np.mean(scores.history['episode_reward']))

    env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', env, verbose = 1)

    model.learn(total_timesteps=60000)

    
