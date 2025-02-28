import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

from flightsim.axes3ds import Axes3Ds
from flightsim.world import World

from proj1_2.code.occupancy_map import OccupancyMap
from proj1_2.code.graph_search import graph_search
# from proj1_2.code.graph_search_2 import graph_search

# Choose a test example file. You should write your own example files too!
# filename = 'test_empty.json'
# filename = 'test_saw.json'
# filename = 'test_impossible.json'
# filename = 'test_forest.json'
#filename = 'test_difficult.json'
filename = 'test_switchback.json'


# Load the test example.
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
resolution = np.ones([3]) * 0.1 # (x,y,z) resolution of discretization, shape=(3,).
margin = 0.725       # Scalar spherical robot size or safety margin.
start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

# Run your code and return the path.
start_time = time.time()
path, node_expanded = graph_search(world, resolution, margin, start, goal, astar=True)
end_time = time.time()

# Print results.
print()
# print('Total number of nodes expanded:', node_expanded)
print(f'Solved in {end_time-start_time:.2f} seconds')
if path is not None:
    number_points = path.shape[0]
    length = np.sum(np.linalg.norm(np.diff(path, axis=0),axis=1))
    print(f'The discrete path has {number_points} points.')
    print(f'The path length is {length:.2f} meters.')
else:
    print('No path found.')

# Draw the world, start, and goal.
fig = plt.figure()
ax = Axes3Ds(fig)
world.draw(ax)
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')

# Plot your path on the true World.
if path is not None:
    world.draw_line(ax, path, color='blue')
    world.draw_points(ax, path, color='blue')

# For debugging, you can visualize the path on the provided occupancy map.
# Very, very slow! Comment this out if you're not using it.
#fig = plt.figure()
# ax = Axes3Ds(fig)
# oc = OccupancyMap(world, resolution, margin)
# oc.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')
# if path is not None:
#     world.draw_line(ax, path, color='blue')
#     world.draw_points(ax, path, color='blue')

'''
fig = plt.figure()
x_astar = np.array([1522099,99757,35749,13040])
x_d = np.array([3141729,198583,71133,23971])
y = np.array([0.2,0.5,0.7,1])
plt.plot(y,np.log(x_astar),label='A*')
plt.plot(y,np.log(x_d),label='Dijkstra')
plt.title('A* vs Dijkstra')
plt.xlabel('Resolution')
plt.ylabel('ln(Nodes Expanded)')
plt.legend()
'''
plt.show()
