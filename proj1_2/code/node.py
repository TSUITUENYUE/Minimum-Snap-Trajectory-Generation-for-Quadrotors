
class Node(object):
    def __init__(self,index, parent, cost, h, astar, is_closed=False):
        self.index = index
        self.parent = parent
        self.cost = cost
        self.heuristic = h
        self.astar = astar
        self.is_closed = is_closed
        if self.astar:
            self.f = self.cost + self.heuristic
        else:
            self.f = self.cost

    def __lt__(self, other):
        return(self.f, self.heuristic) < (other.f, other.heuristic)

    def get_cost(self):
        return self.f