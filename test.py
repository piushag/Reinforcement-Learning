import bisect
import collections
import random, copy, math
import numpy as np
import matplotlib.pyplot as plt

class Mountain_Car(object):
    def _init_(self, nx, nv):

        def get_xv_state(nx, l, r):

            step = (r-l)/nx
            ans = []
            while (l<r):
                l = round(l,4)
                ans.append(l)
                l += step

            ans.append(r)
            return ans

        self.x_state = get_xv_state(nx, -1.2, 0.5)
        self.v_state = get_xv_state(nv, -0.07, 0.07)
        self.end = []
        self.actions = ["R", "N", "F"]

        for v in self.v_state:
            self.end.append([0.5, v])

        #print(self.end)

    def displayState(self):
        print(self.x_state)
        print(self.v_state)

    def getState(self,x,v):
        r = bisect.bisect_right(self.x_state, x)
        c = bisect.bisect_right(self.v_state, v)

        if r==0:
            r = 1

        if c == 0:
            c = 1
        return [self.x_state[r-1], self.v_state[c-1]]


class SARSA_Online(object):

    def TrueOnlineSARSA(self, mountainCar, numEpisode, alpha, lamb, epsilion, gamma, d=10, delta=0.0001):

        policy = collections.defaultdict(set)
        for x in mountainCar.x_state:
            for v in mountainCar.v_state:
                policy[(x,v)].add(mountainCar.actions[random.randint(0,len(mountainCar.actions) - 1)])

        feature_vector = {}
        for x in mountainCar.x_state:
            for v in mountainCar.v_state:
                for a in mountainCar.actions:
                    if [x,v] not in mountainCar.end:
                       feature_vector[(x,v,a)] = np.random.rand(d)
                    else:
                        feature_vector[(x,v,a)] = np.zeros(d)

        for i in range(5000):
            weights = np.zeros(d)
            iter = 0
            numOfActions = 0

            for iter in range(numEpisode):
                x_0, v_0 = random.uniform(-.6, -.4), 0
                tx, tv = mountainCar.getState( x_0, v_0)
                actions, action_prob = self.generate_epsilion_probability(policy[(tx,tv)], mountainCar, epsilion)
                a = np.random.choice(actions, 1, p=action_prob)[0]
                x_vec = feature_vector[(tx,tv, a)]
                z_vec =  np.zeros(d)
                Q_old = 0

                steps = 0
                x,v = x_0, v_0
                while ([tx,tv] not in mountainCar.end and steps<1000):

                    if a == "R":
                        action = -1
                    elif a == "N":
                        action = 0
                    else:
                        action = 1

                    next_v = v + 0.001*action - 0.0025*math.cos(3*x)
                    next_x = x + next_v

                    if next_x <= -1.2:
                        next_v = 0

                    next_tx, next_tv = mountainCar.getState(next_x, next_v)
                    
                    actions, action_prob = self.generate_epsilion_probability(policy[(next_tx,next_tv)], mountainCar, epsilion)
                    next_a = np.random.choice(actions, 1, p=action_prob)[0]
                    next_x_vec = feature_vector[(next_tx, next_tv, next_a)]
                    Q = np.dot(weights,np.transpose(x_vec))
                    Q_next = np.dot(weights,np.transpose(next_x_vec))
                    del_val = -1 + gamma*Q_next - Q
                    z_vec = gamma*lamb*z_vec + (1-alpha*gamma*lamb*np.dot(z_vec,np.transpose(x_vec)))*(x_vec)
                    weights = weights + alpha*(del_val + Q - Q_old)*z_vec - alpha(Q-Q_old)*x_vec
                    Q_old = Q_next
                    feature_vector[(tx,tv,a)] = x_vec
                    x_vec = next_x_vec
                    a = next_a
                    x, v = next_x, next_v
                    steps += 1

            self.updatePolicy(policy, weights, feature_vector)
            self.simulation_mountain_car(policy, mountainCar)
        return policy

    
            
    def updatePolicy(self, policy, weights, feature_vector):
        for x in mountainCar.x_state:
            for v in mountainCar.v_state:
                maxval = float('-inf')
                maxact = ""
                for a in mountainCar.actions:
                    if [x,v] not in mountainCar.end:
                        val = np.dot(weights,np.transpose(feature_vector[(x,v,a)]))
                        if val > maxval:
                            maxval = val
                            maxact = a
                policy[(x,v)] = maxact
        
    def generate_epsilion_probability(self, policy_act, mountainCar, epsilion):
        action_prob = []
        actions = []

        for a in list(policy_act):
            actions.append(a)
            action_prob.append((1-epsilion)/len(policy_act) + epsilion/len(mountainCar.actions))

        for a in mountainCar.actions:
            if a not in actions:
                actions.append(a)
                action_prob.append(epsilion/len(mountainCar.actions))

        return actions, action_prob

    def simulation_mountain_car(self, policy, mountainCar):
        x_0 = random.uniform(-0.6, -0.4)
        v_0 = 0

        x_t, v_t = x_0, v_0
        x_t, v_t = mountainCar.getState(x_t, v_t)
        reward = 0

        while x_t < 0.5 and reward > -1000:
            a = list(policy[(x_t, v_t)])[0]

            if a == "R":
                action = -1
            elif a == "N":
                action = 0
            else:
                action = 1

            v_t = v_t + 0.001*action - 0.0025*math.cos(3*x_t)
            x_t = x_t + v_t

            x_t, v_t = mountainCar.getState(x_t, v_t)

            if x_t <= -1.2: v_t = 0
            reward -= 1

        print(reward)
        return reward


mountainCar = Mountain_Car(17,14)
# mountainCar.displayState()
#print(mountainCar.getState(-2.4,.04))

sarsa = SARSA_Online()
alpha = .2
epsilion = 0
gamma = 1
numEpisode = 20
lamb = .2
policy = sarsa.TrueOnlineSARSA(mountainCar, numEpisode, alpha, lamb, epsilion, gamma)
reward = sarsa.simulation_mountain_car(policy, mountainCar)
print(reward)

# graph_avg = graph_x
# step_avg = step_y
# s = [step_y]

# for i in range(1,20):
#     q_pi, policy, iter, graph_x, graph_y, step_x, step_y = sarsa.SARSA_learning(mountainCar, alpha, epsilion, gamma)
#     graph_avg = [sum(value) for value in zip(graph_avg, graph_y)]
#     step_avg = [sum(value) for value in zip(step_avg, step_y)]
#     s.append(step_y)

# #graph_y = np.array(graph_avg)/20

# plt.plot(graph_x[:len(graph_y)],graph_y)
# plt.xlabel('Number steps')
# plt.ylabel('Number of Episode')
# plt.show()

# std_err = np.std(s,axis=0)
# plt.plot(step_x, step_y)
# plt.fill_between(step_x, np.asarray(step_x) - std_err,
#                  np.asarray(step_x) +  std_err)
# plt.ylabel('Steps to reach Goal')
# plt.xlabel('Number of Episode')
# plt.show()