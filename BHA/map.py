class Map:
    #declare variables
    WALL = "w"
    CORRIDOR = " "
    START = "A"
    GOAL = "X"

    H_UNSET = -1
    STEP_COST = 1


    def __init__(self, filename):
        self.filename = filename
        self.map, self.pos, self.goal = self.load()
        self.h = [[self._manhattan_distance(j, i) for i in range(len(self.map[0]))] for j in range(len(self.map))]
        self.updates = 0
        

    def load(self):
        map = []
        with open(self.filename) as f:
            #reads file line by line
            for line in f:
                map.append(list(line.strip()))
        start, goal = None, None
        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j] == Map.START:
                    start = (i, j)
                if map[i][j] == Map.GOAL:
                    goal = (i, j)
                if start is not None and goal is not None:
                    break
        map[start[0]][start[1]] = Map.CORRIDOR

        return map, start, goal

    def __str__(self):
        map_to_string = "  "
        #adds letters A, B, ... to the len of the first element of the map
        for i in range(len(self.map[0])):
            map_to_string += " " + chr(65 + i) + " "
        map_to_string += "\n"

        for i in range(len(self.map)):
            map_to_string += str(i) + " "
            for j in range(len(self.map[i])):
                if (i, j) == self.pos:
                    map_to_string += " " + Map.START + " "

                elif self.map[i][j] == Map.WALL:
                    map_to_string += "███"

                elif self.map[i][j] == Map.GOAL:
                    map_to_string += " " + Map.GOAL + " "

                else:
                    map_to_string += " " + str(self.h[i][j])
                    if self.h[i][j] < 10 and self.h[i][j] > -1:
                        map_to_string += " "
            map_to_string += "\n"

        return map_to_string
    
    def is_goal(self):
        return self.pos == self.goal

    def move(self, movement, h):
        if h > self.h[self.pos[0]][self.pos[1]]:
            self.h[self.pos[0]][self.pos[1]] = h
            self.updates += 1

        if movement == "r":
            self.pos = (self.pos[0], self.pos[1]+1)
        elif movement == "d":
            self.pos = (self.pos[0]+1, self.pos[1])
        elif movement == "l":
            self.pos = (self.pos[0], self.pos[1]-1)
        elif movement == "u":
            self.pos = (self.pos[0]-1, self.pos[1])

    
    def position(self):
        # returns the current position, where the row is indicated by numbers and the column by letters
        return f"{chr(65 + self.pos[1])}{self.pos[0]}"

    
    def _manhattan_distance(self, x1, y1):
        if self.map[x1][y1] == self.WALL:
            return self.H_UNSET
        return abs(x1 - self.goal[0]) + abs(y1 - self.goal[1])
    
    def _step(self, x, y):
        # Returns f(x,y) = g(x,y) + h(x,y)
        if self.map[x][y] == self.WALL:
            return None
        if self.map[x][y] == self.CORRIDOR:
            return Map.STEP_COST + self.h[x][y]
        if self.map[x][y] == self.GOAL:
            return Map.STEP_COST + 0
        return None
    
    def forward(self):
        x, y = self.pos
        #Right Down Left Up
        return {"r": self._step(x, y+1), "d": self._step(x+1, y), "l": self._step(x, y-1), "u": self._step(x-1, y)}