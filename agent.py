import numpy as np
class env:
    
    def __init__(self, N, wall_points = 3):
        self.square = np.zeros(shape = (N, N), dtype = np.int8)
        self.N = N
        self.ag = [np.random.randint(1, N - 1) , np.random.randint(0, N - 1)]
        self.square[tuple(self.ag)] = 2
        # select a point randomly (in middle somewhat)
        walls = []
        for _ in range(wall_points):
            i, j = np.random.randint(0, N) , np.random.randint(0, N)
            self.square[i][j] = 1
        # mark wall on side
        for i in range(N):
            for j in range(N):
                if i== 0 or j == 0 or i == N -1 or j == N - 1:
                    self.square[i][j] = 1
          
        # get the agent place
        
        self.food = [np.random.randint(1, N-1),  np.random.randint(1, N-1)]
        self.square[tuple(self.food)] = 3
                    
    def showField(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.square[i][j] == 1:
                    print('*', end = ' ')
                elif tuple(self.ag) == (i , j):
                    print('M', end = ' ')
                elif tuple(self.food) == (i , j):
                    print("F", end = ' ')
                else:
                    print(' ', end = ' ')
            print()

    def getFeature(self):
        '''
        features = [threat_up, threat_right, threat_down, threat_left, Is_food_up, Is_food_right, Is_food_down, Is_food_left]  # features, done, reward
        '''
        if self.square[tuple(self.ag)] == 3: # eaten the food
            return [self.square[self.ag[0] - 1, self.ag[1]] == 1, self.square[self.ag[0], self.ag[1] + 1] == 1,
                             self.square[self.ag[0] + 1, self.ag[1]] == 1, self.square[self.ag[0], self.ag[1] - 1] == 1,
                             self.ag[0] > self.food[0], self.food[1] > self.ag[1], self.ag[0] < self.food[0], 
                             self.food[1] <self.ag[1]], 1, 10
        elif self.square[tuple(self.ag)] == 1:
            return [True, True, True, True, False, False, False, False],  1, -1
        else:
            return  [self.square[self.ag[0] - 1, self.ag[1]] == 1, self.square[self.ag[0], self.ag[1] + 1] == 1,
                             self.square[self.ag[0] + 1, self.ag[1]] == 1, self.square[self.ag[0], self.ag[1] - 1] == 1,
                             self.ag[0] > self.food[0], self.food[1] > self.ag[1], self.ag[0] < self.food[0], 
                             self.food[1] <self.ag[1]], 0, 0        
        
    def step(self, action):
        '''
        Do the action
        '''
        if action == 0: # up mve
            self.ag[0] -= 1
        elif action == 1:
            self.ag[1] += 1
        elif action == 2:
            self.ag[0] += 1
        elif action == 3:
            self.ag[1] -= 1
        
        