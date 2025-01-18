from map import Map
class LRTA:
    def __init__(self, filename):
        self.map_name = filename
        self.hard_reset()


    def hard_reset(self):
        self.map = Map(self.map_name)
        self.cum_steps = 0
        self.start = self.map.pos
        self.reset()

    def reset(self):
        self.steps = 0
        self.map.pos = self.start
        self.map.updates = 0
        self.path = [self.map.position()]
        self.actions = []

    def forward(self):
        #Returns true in case the agent is still running
        
        self.steps += 1
        self.cum_steps += 1

        next_moves = self.map.forward()

        valid_moves = [i for i in next_moves.values() if i is not None]
        ordered_moves = sorted(valid_moves)

        min_movement = ordered_moves[0]

        for move in ["r","d","l","u"]:
            if next_moves[move] == min_movement:
                self.map.move(move, min_movement)
                self.actions.append(move)
                break

        self.path.append(self.map.position())

        if self.map.is_goal():
            return False
        return True
    
    def __str__(self):
        return str(self.map)