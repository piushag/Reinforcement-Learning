from collections import defaultdict
import random
import  numpy as np
import matplotlib.pyplot as plt
import math

states=[]
for i in range(5):
    for j in range(5):
        states.append((i,j))


states_map = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15], [16,17,18,19,20],[21,22,23,24,25]]

states_initial=[]
for i in range(5):
    for j in range(5):
        states_initial.append((i,j))

states_initial.remove((4,4))

gamma = 0.9
water = (4,2)
goal = (4, 4)
obstacles = [(2,2), (3,2)]

for obstacle in obstacles:
    states.remove(obstacle)
    states_initial.remove(obstacle)


actions = ['>', '<', '^', 'v']

actions_map = [0, 1, 2, 3]


action_values = defaultdict(dict)
action_values = {
    '>': (0,1), '<': (0,-1),  '^': (-1,0), 'v': (1,0)
}

action_possibilities = defaultdict(dict)
action_possibilities = {
    '>': ('v','^'), '<': ('^','v'),  '^': ('>','<'), 'v': ('<','>')
}

policy = {}
for state in states:
    if state == goal:
        policy[state] = 'G'
    elif state in obstacles:
        policy[state] = 0
    else:
        policy[state] = random.choice(actions)



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




rewards = {}
for state in states:
    if state == water:
        rewards[state] = -10
    elif state == goal:
        rewards[state] = 10
    else:
        rewards[state] = 0


def create_p_values(state, action):
    next_states = []
    state_dist = []
    vals = action_values.get(action, 0)
    poss = action_possibilities.get(action, 0)


    next_possible_state_1 = (state[0]+vals[0], state[1]+vals[1])
    if(next_possible_state_1 in states):
        next_states.append(next_possible_state_1)
        state_dist.append(0.8)
        next_states.append(state)
        state_dist.append(0.1)
    else:
        next_states.append(state)
        state_dist.append(0.9)


    for act in [poss[0], poss[1]]:
        next_vals = action_values.get(act, 0)
        next_possible_state_2 = (state[0]+next_vals[0], state[1]+next_vals[1])
        if(next_possible_state_2 in states):
            next_states.append(next_possible_state_2)
            state_dist.append(0.05)
        else:
            state_dist[next_states.index(state)]+=0.05

    return next_states, state_dist



def generate_episode(alpha, lambd, weights, epsilon):

    episode = []
    
    comp = True

    state = random.choice(states_initial)

    x = random.random()

    action = getAction(x, epsilon, weights)

    z = np.zeros((1, len(actions)))

    q_old = 0
   

    while comp:
        episode.append(state)

        s1, state_dist = create_p_values(state, action)
        next_state =s1[np.random.choice(len(s1), 1, p=state_dist)[0]]

        reward = rewards[next_state]

        next_action = ''

        delt = 0
        q_next = 0

        if next_state != goal:

            x_next = random.random()

            next_action = getAction(x_next, epsilon, weights)

            q = np.dot(np.transpose(weights[:,  actions_map[actions.index(action)]]),x)

            q_next = np.dot(np.transpose(weights[:,  actions_map[actions.index(next_action)]]),x_next)

            delt = reward + gamma*q_next - q



        else:
            x_next = 0

            q = np.dot(np.transpose(weights[:,  actions_map[actions.index(action)]]),x)

            q_next = 0

            delt = reward + gamma*q_next - q

            comp = False
        
        z[:,  actions_map[actions.index(action)]] = gamma*lambd*z[:,  actions_map[actions.index(action)]] + (1 - alpha*gamma*lambd*np.dot(np.transpose(z[:, actions_map[actions.index(action)]]), x))*(x)

        weights[:, actions_map[actions.index(action)]] = weights[:, actions_map[actions.index(action)]] + alpha*(delt + q - q_old)*z[:, actions_map[actions.index(action)]] - alpha*(q - q_old)*x

        q_old = q_next

        x = x_next
        
        
        state = next_state
        action = next_action

    return episode 


number_of_episodes = 200
c_arr = [i for i in range(1,number_of_episodes+1)]
def generate_learning_curve():

    states_count = []
    states_num = 0

    states_each = []


    alpha = 0.5
    go = True
    epsilon = 0.5

    lambd = 0.1

    weights = np.zeros((1, len(actions)))

    for c in range(number_of_episodes):
        states_epi = generate_episode(alpha, lambd, weights, epsilon)
        print(c)
        states_num = states_num+len(states_epi)
        states_count.append(states_num)
        states_each.append(len(states_epi))
                
        if c%10 == 0:
            if epsilon - 0.05>0:
                epsilon = epsilon-0.05

    return states_count, states_each


time_steps_arr = [0 for _ in range(number_of_episodes)] 
each_ct_arr = [0 for _ in range(number_of_episodes)] 

for t in range(20):

    states_count, states_each = generate_learning_curve()


    for l in range(number_of_episodes):
        time_steps_arr[l] += states_count[l]
        each_ct_arr[l] += states_each[l]


for l in range(number_of_episodes):
    time_steps_arr[l] = time_steps_arr[l]/20   
    each_ct_arr[l] += states_each[l]/20

x_arr = [i for i in range(1,number_of_episodes+1)]

plt.plot(x_arr, each_ct_arr)
plt.xlabel('Number of Episodes')
plt.ylabel('Steps')
plt.show()
    


plt.plot(time_steps_arr, c_arr)
plt.xlabel('Time Steps')
plt.ylabel('Episodes')
plt.show()








