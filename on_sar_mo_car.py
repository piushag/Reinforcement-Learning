import numpy as np
import random
import math
import matplotlib.pyplot as plt

actions = [-1, 0, 1]

actions_map = [0, 1, 2]

def normaliseS(x, v):
    x = (x+1.2)/1.7
    v = (v+0.07)/0.14
    return x, v

def getFourierBasis(x, v, order):
  x_norm, v_norm  = normaliseS(x, v)

  phi_s = [1]
  for i in range(1,order+1):
    phi_s.append(math.cos(i*math.pi*x_norm))
  for i in range(1,order+1):
    phi_s.append(math.cos(i*math.pi*v_norm))

  return phi_s


def getAction(x, epsilon, weights):
    if random.random() < epsilon:
        action = random.choice(actions)
    else: 
        vals = []
        for a in actions:
            vals.append(np.dot(np.transpose(weights[:, actions_map[actions.index(a)]]), x))

        q_pi = np.argmax(vals)
        action = actions[actions_map.index(q_pi)]
    
    return action



def generateEpisode(epsilon, alpha, gamma, lamb, weights, order):
    x_t = np.random.uniform(-0.6, -0.4)
    v_t = 0

    z = np.zeros((2*order+1, len(actions)))

    c = 0

    q_old = 0

    x = np.array(getFourierBasis(x_t, v_t, order))
    
    action = getAction(x, epsilon, weights)

    while x_t < 0.5 and c<1000:

        v_t_next = v_t + 0.001*action  - 0.0025*math.cos(3*x_t)
        x_t_next = x_t + v_t_next

        if x_t_next == -1.2: 
            v_t_next = 0

        x_t_next = max(-1.2, x_t_next)
        x_t_next = min(0.5, x_t_next)

        v_t_next = max(-0.07, v_t_next)
        v_t_next = min(0.07, v_t_next)

        if x_t_next == 0.5:
            v_t_next = 0
            reward = 0
        else:
            reward = -1

        x_new = np.array(getFourierBasis(x_t_next, v_t_next, order))

        q = np.dot(np.transpose(weights[:, actions_map[actions.index(action)]]),x)

        action_next = getAction(x_new, epsilon, weights)

        q_next = np.dot(np.transpose(weights[:, actions_map[actions.index(action_next)]]), x_new)

        delt = reward + gamma*q_next - q

        z[:, actions_map[actions.index(action)]] = gamma*lamb*z[:, actions_map[actions.index(action)]] + (1-alpha*gamma*lamb*np.dot(np.transpose(z[:, actions_map[actions.index(action)]]), x))*(x)

        weights[:, actions_map[actions.index(action)]] = weights[:, actions_map[actions.index(action)]] + alpha*(delt + q - q_old)*z[:, actions_map[actions.index(action)]] - alpha*(q - q_old)*x

        q_old = q_next
                    
        x = x_new

        x_t = x_t_next
        v_t = v_t_next

        action = action_next

        c += 1
   
    
    return c

number_of_epis = 1000
def generate_learning_curve():
    count_arr2 = []
    count_arr = []
    epi_arr = []
    counts2 = 0
    order = 3
    alpha = 0.1
    weights = np.zeros((2*order+1, len(actions)))
    lamb = 0.1
    gamma = 1
    epsilon = 1
    for epi in range(number_of_epis):
        counts = generateEpisode(epsilon, alpha, gamma, lamb, weights, order)
        counts2 += counts
        count_arr2.append(counts2)
        count_arr.append(counts)
        epi_arr.append(epi)
        if epi%20 == 0 and epsilon-0.05>0:
            epsilon -= 0.05

    return count_arr, count_arr2


all_count_arr = [0 for _ in range(number_of_epis)] 
all_epi_arr = [i for i in range(number_of_epis)] 


all_count_arr2 = [0 for _ in range(number_of_epis)] 


std_lists = []
for i in range(number_of_epis):
    std_lists.append([])

for l in range(20):
    count_arr, counts_arr2 = generate_learning_curve()

    for i in range(number_of_epis):
        all_count_arr[i] += count_arr[i]
        all_count_arr2[i] += counts_arr2[i]
        std_lists[i].append(count_arr[i])

std = []
for i in range(number_of_epis):
    std.append(np.std(std_lists[i]))



for i in range(number_of_epis):
        all_count_arr[i] = all_count_arr[i]/20
        all_count_arr2[i] = all_count_arr2[i]/20
    

plt.plot(all_epi_arr, all_count_arr, color = 'black')
plt.errorbar(all_epi_arr, all_count_arr, std, linestyle='None', marker='|', ecolor='blue')
plt.ylabel('Time steps')
plt.xlabel('Episodes')
plt.show()

plt.plot(all_count_arr2, all_epi_arr)
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.show()

