
import collections
import numpy as np
import random


class gridWorld(object):

    def __init__(self, dimension):
        self.dimension = dimension
        self.dirs = [] # NESW
        self.dir_ratios = []
        self.actions = []
        self.directions = {}
        self.reward = collections.defaultdict(int)
        self.blockState = set()
        self.start, self.end = [], []
        self.gamma = 1
        self.explore_Reward = collections.defaultdict(float)
        self.visit_count = collections.defaultdict(int)
        self.transition_prob = collections.defaultdict(float)
        self.policy = {
            (0, 0): 'AR', (0, 1): 'AR', (0, 2): 'AR', (0, 3): 'AD', (0, 4): 'AD',
            (1, 0): 'AR', (1, 1): 'AR', (1, 2): 'AR', (1, 3): 'AD', (1, 4): 'AD', 
            (2, 0): 'AU', (2, 1): 'AU', (2, 3): 'AD', (2, 4): 'AD', 
            (3, 0): 'AU', (3, 1): 'AU', (3, 3): 'AD', (3, 4): 'AD', 
            (4, 0): 'AU', (4, 1): 'AU', (4, 2): 'AR', (4, 3): 'AR'
            }


    def setDirs(self, dirs):
        self.dirs = dirs

    def setActions(self, actions):
        self.actions = actions

    def setDirRatios(self, ratios):
        self.dir_ratios = ratios

    def setDirections(self, directions):
        self.directions = directions

    def setGamma(self, gamma):
        self.gamma = gamma

    def setReward(self, cur_x, cur_y, action, next_x, next_y, value):
        self.reward[(cur_x, cur_y, action, next_x, next_y)] = value

    def displayReward(self):
        return (self.reward)

    def setBlockState(self, x_cor, y_cor):
        self.blockState.add((x_cor, y_cor))

    def setStartEnd(self, start, end):
        self.start, self.end = start, end
       
    def displayValueFunction(self,  state_value):
        print(f'New Value Function:')
        for r in range(len(state_value)):
            for c in range(len(state_value[0])):
                if state_value[r][c]  == float('-inf'):
                    state_value[r][c] = 0
                print("%.4f" % state_value[r][c], end="   ")
            print()

    def displayPolicy(self, policy):
        print(f'New Policy:')
        display_policy = self.getPolicy(policy)
        for ele in display_policy:
            for e in ele:
                print(e, end="   ")
            print()

    def getPolicy(self, policy):
        policy_actions= [[" "]*(self.dimension) for _ in range(self.dimension)]

        for r in range(self.dimension):
            for c in range(self.dimension):
                
                if [r,c] in self.end:
                    policy_actions[r][c] = "G"
                    continue

                if (r,c) in self.blockState:
                    policy_actions[r][c] = " "
                    continue

                policy_actions[r][c] = self.directions[policy[(r,c)]]

        return policy_actions

    def generateEpisode(self, policy, num_of_iteration, explore): # [[r,c,a,r_next,c_next,reward], [r_next,c_next,a',r_next',c_next',reward'],..]

        r_0, c_0 = random.randint(0,self.dimension-1), random.randint(0,self.dimension-1)

        while (r_0, c_0) in self.blockState or [r_0, c_0] in self.end:
            r_0, r_0 = random.randint(0,self.dimension-1), random.randint(0,self.dimension-1)

        r, c = r_0, c_0
        episode = []

        i = 0
        #while ([r,c] not in self.end and i<10*num_of_iteration):
        while ([r,c] not in self.end):
            a = policy[(r,c)]
            if a == "AU":
                next = [[r-1,c],[r,c-1], [r,c+1],[r,c]]
            elif a == "AR":
                next = [[r,c+1],[r-1,c], [r+1,c],[r,c]]
            elif a == "AD":
                next = [[r+1,c],[r,c+1], [r,c-1],[r,c]]
            elif a == "AL":
                next = [[r,c-1],[r+1,c], [r-1,c],[r,c]]

            next_r, next_c = next[np.random.choice([0,1,2,3], 1, p = [.8,.05,.05,.1])[0]]

            if (next_r, next_c) in self.blockState or next_r not in range(self.dimension) or next_c not in range(self.dimension):
                next_r, next_c = r, c

            episode.append([r,c,a,next_r, next_c, self.reward.get((r,c,a,next_r, next_c),0)])
            self.explore_Reward[(r,c,a,next_r, next_c)] = self.reward.get((r,c,a,next_r, next_c),0)
            self.visit_count[(r,c,a,next_r, next_c)] += 1
            i += 1
            r , c = next_r, next_c

        return episode

    def explore(self, num_episode):
        
        for _ in range(num_episode):
            self.generateEpisode(self.policy, num_episode, True)

    def calculate_p(self):
        temp = {}
        for r in range(self.dimension):
            for c in range(self.dimension):
                for a in self.actions:
                    cnt = 0
                    for next_r in range(self.dimension):
                        for next_c in range(self.dimension):
                            cnt += self.visit_count[(r,c,a,next_r, next_c)]
                    temp[(r,c,a)] = cnt

        for r in range(self.dimension):
            for c in range(self.dimension):
                for a in self.actions:
                    for next_r in range(self.dimension):
                        for next_c in range(self.dimension):
                            if (r,c,a) in temp and temp[(r,c,a)]!=0:
                                x = self.visit_count[(r,c,a,next_r, next_c)]/temp[(r,c,a)]
                                if x > 0:
                                    self.transition_prob[(r,c,a,next_r, next_c)] = round(x,4)
        return self.transition_prob, self.explore_Reward

    def getMaxAction(self, r, c, state_values):
        max_val, maxpolicy = float('-inf'), self.directions["AU"]

        for i, a in enumerate(self.actions):
            temp_val = 0
            for j, prob in self.dir_ratios:
                next_r, next_c = r + self.dirs[(i+j)%4][0], c + self.dirs[(i+j)%4][1]

                if next_r in range(self.dimension) and next_c in range(self.dimension) and (next_r, next_c) not in self.blockState:
                    temp_val += self.transition_prob[(r,c,a,next_r, next_c)] * (self.reward.get((r,c,a,next_r, next_c), 0) + self.gamma* state_values[next_r][next_c])
                else:
                    temp_val += self.transition_prob[(r,c,a,r,c)] * (self.reward.get((r,c,a,r,c), 0) + self.gamma* state_values[r][c])

            temp_val += self.transition_prob[(r,c,a,r,c)] * (self.reward.get((r, c, a, r, c), 0) + self.gamma* state_values[r][c])

            if temp_val > max_val:
                max_val = temp_val
                maxpolicy = self.directions[a]

        return max_val, maxpolicy

    def value_iteration(self, num_of_iteration, epsilion = 0.0001):

        v_0 =  [[0]*(self.dimension) for _ in range(self.dimension)]
        v = v_0
        for iter in range(num_of_iteration):
            delta = 0
            import copy
            v_old = copy.deepcopy(v)
            for r in range(self.dimension):
                for c in range(self.dimension):
                    if [r,c] in self.end or (r,c) in self.blockState:
                        v[r][c] = 0
                        continue

                    max_val, _ = self.getMaxAction(r, c, v_old)
                    delta = max(delta, abs(max_val - v[r][c]))
                    v[r][c] = max_val
                    
            if delta < epsilion:
                return v, iter

        return v, iter

    def monteCarloFirstVisit(self, num_of_iteration, epsilion = 0.0001):
        q_pi_list = collections.defaultdict(list)
        q_pi = collections.defaultdict(float)
        v_pi = [[float('-inf')]*(self.dimension) for _ in range(self.dimension)]
        policy = {}


        def generateEpisode_based_on_obs(policy, num_of_iteration): # [[r,c,a,r_next,c_next,reward], [r_next,c_next,a',r_next',c_next',reward'],..]

            r_0, c_0 = random.randint(0,self.dimension-1), random.randint(0,self.dimension-1)

            while (r_0, c_0) in self.blockState or [r_0, c_0] in self.end:
                r_0, r_0 = random.randint(0,self.dimension-1), random.randint(0,self.dimension-1)

            r, c = r_0, c_0
            episode = []

            i = 0
            #while ([r,c] not in self.end and i<10*num_of_iteration):
            while ([r,c] not in self.end):
                a = policy[(r,c)]
                if a == "AU":
                    next = [[r-1,c],[r,c-1], [r,c+1],[r,c]]
                    pp = [self.transition_prob[(r,c,a,r-1,c)], self.transition_prob[(r,c,a,r,c-1)], self.transition_prob[(r,c,a,r,c+1)], self.transition_prob[(r,c,a,r,c)]]
                elif a == "AR":
                    next = [[r,c+1],[r-1,c], [r+1,c],[r,c]]
                    pp = [self.transition_prob[(r,c,a,r,c+1)], self.transition_prob[(r,c,a,r-1,c)], self.transition_prob[(r,c,a,r+1,c)], self.transition_prob[(r,c,a,r,c)]]
                elif a == "AD":
                    next = [[r+1,c],[r,c+1], [r,c-1],[r,c]]
                    pp = [self.transition_prob[(r,c,a,r+1,c)], self.transition_prob[(r,c,a,r,c+1)], self.transition_prob[(r,c,a,r,c-1)], self.transition_prob[(r,c,a,r,c)]]
                elif a == "AL":
                    next = [[r,c-1],[r+1,c], [r-1,c],[r,c]]
                    pp = [self.transition_prob[(r,c,a,r,c-1)], self.transition_prob[(r,c,a,r+1,c)], self.transition_prob[(r,c,a,r-1,c)], self.transition_prob[(r,c,a,r,c)]]

                psum = 1-sum(pp)
                pp[0] += psum
                next_r, next_c = next[np.random.choice([0,1,2,3], 1, p = pp)[0]]

                if (next_r, next_c) in self.blockState or next_r not in range(self.dimension) or next_c not in range(self.dimension):
                    next_r, next_c = r, c

                episode.append([r,c,a,next_r, next_c, self.reward.get((r,c,a,next_r, next_c),0)])
                i += 1
                r , c = next_r, next_c

            return episode

        def calculateReward(episode): #{(r,c,a):value}
            ans = {}
            visit = set()

            dp = {}
            dp[len(episode) - 1] = episode[len(episode) - 1][5]

            for i in range(len(episode) - 2, -1, -1):
                dp[i] = episode[i][5] + self.gamma*dp[i+1]

            for i in range(len(episode)):
                r,c,a,r_next,c_next,reward = episode[i]
                if (r,c,a) in visit:
                    continue

                visit.add((r,c,a))
                ans[(r,c,a)] = dp[i]

            return ans


        for i in range(num_of_iteration):
            episode = generateEpisode_based_on_obs(self.policy, num_of_iteration)
            #print(episode)

            G_values = calculateReward(episode)
            #print(G_values)

            for s, v in G_values.items():
                q_pi_list[s].append(v)
                q_pi[s] = np.mean(q_pi_list[s])

            temp_set = set()

            for k in G_values.keys():
                temp_set.add((k[0],k[1]))

            for r in range(self.dimension):
                for c in range(self.dimension):
                    maxVal = v_pi[r][c]
                    for a in self.actions:
                        if (r,c) in temp_set:
                            if q_pi[(r,c, a)] >=  maxVal:
                                maxVal = q_pi[(r,c,a)]
                                policy[(r,c)] = a
                    v_pi[r][c] = maxVal
            #self.displayValueFunction(v_pi)
            #self.displayPolicy(policy)

        return policy, q_pi, v_pi

        




if __name__ == "__main__":

    #Q1
    gamma = 0.9
    start = [0, 0]
    goal = [[4,4]]
    rewards = [[goal[0][0], goal[0][1], 10], [4,2,-10]]

    grid_world = gridWorld(5)
    grid_world.setGamma(gamma)

    #set actions and directions:
    grid_world.setActions(["AU", "AR", "AD", "AL"])
    grid_world.setDirs([[-1, 0], [0, 1], [1, 0], [0, -1]])
    grid_world.setDirRatios([[-1,.05], [0, .8], [1,.05]])
    grid_world.setDirections({"AU":"^","AR":">","AD":"v","AL":"<"})

    grid_world.setBlockState(2,2)
    grid_world.setBlockState(3,2)

    grid_world.setStartEnd(start, goal)
    
    #set Reward
    for reward_x, reward_y, points in rewards:
        for x_cor in range(grid_world.dimension):
            for y_cor in range(grid_world.dimension):
                for a in grid_world.actions:
                    grid_world.setReward(x_cor, y_cor, a, reward_x, reward_y, points)

    #value iteration
    grid_world.explore(50000)
    p, R = grid_world.calculate_p()
    print("Estimated trasition Function")
    print(p)
    print(" ")
    print("Estimated Reward function")
    print(R)
    #policy, q_pi, v_pi = grid_world.monteCarloFirstVisit(50000)
    #v_pi, iter = grid_world.value_iteration(10000)
    grid_world.displayValueFunction(v_pi)
    #print(p)
    #print(p)
    #grid_world.displayValueFunction(v_pi)
    # print(f'Gamma: {gamma}')
    # print(f'iteration: {iter}')
    # print(f'State Value:')
    # for ele in state_value:
    #     for e in ele:
    #         print("%.4f" % e, end="   ")
    #     print()

    # print()
    # print(f'Policy:')
    # optimal_policy = grid_world.getPolicy(policy)
    # for ele in optimal_policy:
    #     for e in ele:
    #         print(e, end="   ")
    #     print()
