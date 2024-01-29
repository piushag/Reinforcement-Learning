from collections import defaultdict
import random
import  numpy as np
import matplotlib.pyplot as plt

states=[]
for i in range(5):
    for j in range(5):
        states.append((i,j))

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


action_values = defaultdict(dict)
action_values = {
    '>': (0,1), '<': (0,-1),  '^': (-1,0), 'v': (1,0)
}

action_possibilities = defaultdict(dict)
action_possibilities = {
    '>': ('v','^'), '<': ('^','v'),  '^': ('>','<'), 'v': ('<','>')
}

v_optimal = [
[4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
[4.3716,  5.0324,  5.8013,  6.6473,  7.3907],
[3.8672,  4.3900,  0.0000,  7.5769,  8.4637],
[3.4182,  3.8319,  0.0000,  8.5738,  9.6946],
[2.9977,  2.9309,  6.0733,  9.6946,  0.0000]
]



policy_arr =[['>',  '>' , '>' , 'v'  ,'v'],['>' , '>' , '>' , 'v' ,'v'],['^', '^' , '' ,'v' , 'v'],['^' , '^' ,'', 'v'  ,'v'],['^' , '^'  ,'>' , '>'  ,'G']]



policy = {}
for state in states:
    if state == goal:
        policy[state] = 'G'
    elif state in obstacles:
        policy[state] = 0
    else:
        policy[state] = random.choice(actions)


policy_init = [[0 for _ in range(5)] for _ in range(5)]
for state in states:
    policy_init[state[0]][state[1]] = policy[state]

mx = 6
print('Initial Policy:')
for row in policy_init:
    print("  ".join(["{:<{mx}}".format(ele,mx=mx) for ele in row]))



pi_values = {}
for state in states:
    pi_values[state, policy[state]] = 1
    for act in actions:
        if act != policy[state]:
            pi_values[state, act] = 0


def update_policy(eps):
    for state in states:
        pi_values[state, policy[state]] = 1 - eps + eps/4
        for act in actions:
            if act != policy[state]:
                pi_values[state, act] = eps/4



def get_action_dist(state):
    dists = []
    acts = []
    for action in actions:
        acts.append(action)
        dists.append(pi_values[state, action])
    
    return acts, dists






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



def generate_episode(alpha, q_func, e_func, lambd):

    episode = []
    
    comp = True

    state = random.choice(states_initial)

    acts, acts_dist = get_action_dist(state)
    action =acts[np.random.choice(len(acts), 1, p=acts_dist)[0]]
   

    while comp:
        episode.append(state)

        s1, state_dist = create_p_values(state, action)
        next_state =s1[np.random.choice(len(s1), 1, p=state_dist)[0]]

        next_action = ''

        reward = rewards[next_state]

        delt = 0

        if next_state != goal:
            acts2, acts_dist2 = get_action_dist(next_state)

            next_action =acts2[np.random.choice(len(acts2), 1, p=acts_dist2)[0]]

            delt = reward + gamma*q_func[next_state, next_action] - q_func[state, action]

        else:
            delt = reward - q_func[state, action]
            comp = False
        
        
        e_func[state, action] = e_func[state, action]+1

        q_func[state, action] = q_func[state, action] + alpha*delt*e_func[state, action]

        e_func[state, action] = gamma*lambd*e_func[state, action]
        
        state = next_state
        action = next_action

    return episode 

c_arr = [i for i in range(1,201)]
def sars():

    # x = [i for i in range(1,201)]
    y = []
    values  = [[0 for _ in range(5)] for _ in range(5)]

    states_count = []
    states_num = 0

    q_func = {}

    for state in states:
        for act in actions:
            q_func[state, act] = 0


    e_func = {}

    for state in states:
        for act in actions:
            e_func[state, act] = 0





    alpha = 0.099
    go = True
    epsilon = 0.5
    lambd = 0.5

    #c = 0

    #while go:
    for c in range(200):
        states_epi = generate_episode(alpha, q_func, e_func, lambd)
        states_num = states_num+len(states_epi)
        states_count.append(states_num)

        
        for state in states_epi:
            q_arr = []
            a_arr = []
            for act in actions:
                q_arr.append(q_func[state, act])
                a_arr.append(act)
                
            q_max = max(q_arr)
            act_max = a_arr[q_arr.index(q_max)]
            policy[state] = act_max
                
        if c%10 == 0:
            if epsilon - 0.05>0:
                epsilon = epsilon-0.05

        update_policy(epsilon)

        values_2  = [[0 for _ in range(5)] for _ in range(5)]
        for state in states:
            for act in actions:
                if state == goal or state in obstacles:
                    values_2[state[0]][state[1]] = 0
                else:
                    values_2[state[0]][state[1]] += pi_values[state, act] * q_func[state, act]

        sum = 0
        for state in states:
            sum += pow(v_optimal[state[0]][state[1]] - values_2[state[0]][state[1]], 2)

        y.append(sum/25)

        # delta = 0
        # for state_v in states:
        #     delta = max(abs(v_optimal[state_v[0]][state_v[1]] - values_2[state_v[0]][state_v[1]]), delta)
        # if delta < 0.5:
        #     go = False

        #print(c)
        #c+=1

    
    
    for state in states:
        for act in actions:
            if state == goal or state in obstacles:
                values[state[0]][state[1]] = 0
            else:
                values[state[0]][state[1]] += pi_values[state, act] * q_func[state, act]

    
    print('\n')
    print('New Policy:')

    policy_small = [[0 for _ in range(5)] for _ in range(5)]
    for state in states:
        policy_small[state[0]][state[1]] = policy[state]

    mx = 6

    for row in policy_small:
        print("  ".join(["{:<{mx}}".format(ele,mx=mx) for ele in row]))

    print('\n')
    print('Value Function:')
    mx = 6

    for row in values:
        print("  ".join(["{:.4f}".format(ele,mx=mx) for ele in row]))

    # plt.plot(x, y)
    # plt.xlabel('Number of Episodes')
    # plt.ylabel('MSE')
    # plt.show()


    return states_count, y




time_steps_arr = [0 for _ in range(200)] 
all_mse_arr = [0 for _ in range(200)] 

for t in range(20):

    states_count, y = sars()


    for l in range(200):
        # print('\n')
        # print(states_count[l])
        time_steps_arr[l] += states_count[l]
        all_mse_arr[l] += y[l]
        # print(time_steps_arr[l])


# sub = 0.8
# for l in range(1,200):
#     time_steps_arr[l] = time_steps_arr[l] + time_steps_arr[l-1]
#     time_steps_arr[l] = time_steps_arr[l] - sub
#     sub = sub + 0.8

for l in range(200):
    time_steps_arr[l] = time_steps_arr[l]/20   

x_arr = [i for i in range(1,201)]

plt.plot(x_arr, all_mse_arr)
plt.xlabel('Number of Episodes')
plt.ylabel('MSE')
plt.show()
    


plt.plot(time_steps_arr, c_arr)
plt.xlabel('Time Steps')
plt.ylabel('Episodes')
plt.show()








