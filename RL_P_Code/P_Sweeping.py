import collections, heapq
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

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


class prioritizedSweeping(object):
    
    def pritoritzed_sweeping(self, grid_world, numIterations, n_value, alpha = 0.2, theta = .5):

        q_pi = {}
        model = {}
        policy = {}

        for r in range(grid_world.dimension):
            for c in range(grid_world.dimension):
                policy[(r,c)] = grid_world.actions[random.randint(0,3)]

        #["AU", "AR", "AD", "AL"]
        dirs = [[-1,0], [0,1], [1,0],[0,-1]]

        for index, a in enumerate(grid_world.actions):
            for i in range(grid_world.dimension):
                for j in range(grid_world.dimension):
                    if (i,j) not in grid_world.blockState and [i,j] not in grid_world.end:
                        q_pi[(i,j,a)] = random.random()
                        dr, dc = dirs[index]
                        r, c = i+dr, j+dc

                        if (r,c) in grid_world.blockState or r not in range(grid_world.dimension) or c not in range(grid_world.dimension):
                            r,c = i, j

                        model[(i,j,a)] = [r,c,grid_world.reward.get((i,j,a,r,c),0)]
                    else:
                        q_pi[(i,j,a)] = 0

        pq = []

        for iter in range(numIterations):
            r_0, c_0 = random.randint(0,grid_world.dimension-1), random.randint(0,grid_world.dimension-1)

            while (r_0, c_0) in grid_world.blockState or [r_0, c_0] in grid_world.end:
                r_0, r_0 = random.randint(0,grid_world.dimension-1), random.randint(0,grid_world.dimension-1)

            r, c = r_0, c_0
            a = policy[(r,c)]

            if a == "AU":
                next = [r-1,c]
            elif a == "AR":
                next = [r,c+1]
            elif a == "AD":
                next = [r+1,c]
            elif a == "AL":
                next = [r,c-1]

            next_r, next_c = next[0], next[1]

            if (next_r, next_c) in grid_world.blockState or next_r not in range(grid_world.dimension) or next_c not in range(grid_world.dimension):
                next_r, next_c = r, c

            reward = grid_world.reward.get((r,c,a,next_r, next_c),0)
            model[(r,c,a)] = [next_r, next_c, reward]

            maxval = float('-inf')
            for act in grid_world.actions:
                if q_pi[(next_r,next_c,act)] > maxval:
                    maxval =  q_pi[(next_r, next_c, act)]

            pValue = abs(reward + grid_world.gamma*maxval - q_pi[(r,c,a)])

            if pValue > theta:
                heapq.heappush(pq,[-pValue, r, c, a])

            maxiter = 0
            while pq and maxiter < n_value:
                maxiter += 1
                _, r, c, a = heapq.heappop(pq)
                next_r, next_c, reward = model[(r,c,a)]
                q_pi[(r,c,a)] = q_pi[(r,c,a)] + alpha*(reward + grid_world.gamma*maxval - q_pi[(r,c,a)])

                for dr, dc in dirs:
                    r_, c_ = r+dr, c+dc
                    if (r_,c_) in grid_world.blockState or r_ not in range(grid_world.dimension) or c_ not in range(grid_world.dimension) or [r_,c_] in grid_world.end:
                        continue

                    for a in grid_world.actions:
                        predReward = grid_world.reward.get((r_,c_,a,r,c), 0)
                        maxval = float('-inf')
                        for act in grid_world.actions:
                            if q_pi[(r,c,act)] > maxval:
                                maxval = q_pi[(r,c,act)]

                        pValue = abs(predReward+ grid_world.gamma*maxval - q_pi[(r_,c_,a)])

                        if pValue > theta:
                            heapq.heappush(pq, [-pValue, r_,c_,a])

        return q_pi

    def generate_value_function(self, q_pi, grid_world):

        v_pi = [[0]*(grid_world.dimension) for _ in range(grid_world.dimension)]
        policy =  [[" "]*(grid_world.dimension) for _ in range(grid_world.dimension)]
        for r in range(grid_world.dimension):
            for c in range(grid_world.dimension):
                maxval = float('-inf')
                maxact = ""
                for act in grid_world.actions:
                    if q_pi[(r,c,act)] > maxval:
                        maxval = q_pi[(r,c,act)]
                        maxact = act

                v_pi[r][c] = maxval
                policy[r][c] = maxact

        return v_pi, policy

    def displayValueFunction(self,  state_value):
        print(f'Value Function:')
        for r in range(len(state_value)):
            for c in range(len(state_value[0])):
                if state_value[r][c]  == float('-inf'):
                    state_value[r][c] = 0
                print("%.4f" % state_value[r][c], end="   ")
            print()

if __name__ == "__main__":
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

    psweeping = prioritizedSweeping()
    q_pi = psweeping.pritoritzed_sweeping(grid_world, 10000, 10000)
    v_pi, optimal_policy = psweeping.generate_value_function(q_pi, grid_world)
    psweeping.displayValueFunction(v_pi)

    print(optimal_policy)
    for r in range(len(optimal_policy)):
        for c in range(len(optimal_policy[0])):
            #print(optimal_policy[r][c])
            if optimal_policy[r][c] == "AU":
                optimal_policy[r][c] = "^"

            if optimal_policy[r][c] == "AD":
                optimal_policy[r][c] = "v"

            if optimal_policy[r][c] == "AR":
                optimal_policy[r][c] = ">"

            if optimal_policy[r][c] == "AL":
                optimal_policy[r][c] = "<"

            print(optimal_policy[r][c], end="   ")
        print()



    


                            




