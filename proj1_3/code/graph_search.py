from heapq import heappush, heappop  # Recommended.
import numpy as np
from flightsim.world import World
from .occupancy_map import OccupancyMap # Recommended.
def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    nodes_expanded = 0

    six_deltas = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    six_deltas = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
                        (1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                        (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
                        (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1),
                        (1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),
                        (-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]

    parent_map = {}

    map = occ_map.map
    length, width, height = map.shape
    open_list = []
    close_set = set()
    path = []
    idx = []

    gv = np.inf*np.ones((length, width, height))

    def heuristic(index):
        # return 2.0 * (abs(index[0] - goal_index[0]) + abs(index[1] - goal_index[1])+ abs(index[2] - goal_index[2]))
        return ((index[0] - goal_index[0]) ** 2 + (index[1] - goal_index[1]) ** 2 + (index[2] - goal_index[2]) ** 2) ** 0.5
        # return 1.5 * np.linalg.norm(np.array(index) - np.array(goal_index))

    h0 = 0 + heuristic(start_index) if astar else 0
    gv[start_index] = 0
    heappush(open_list, (h0, 0, start_index))
    while open_list:
        f_current,g_current,idx_current = heappop(open_list)
        if idx_current in close_set:
            continue
        close_set.add(idx_current)
        nodes_expanded += 1

        if idx_current == goal_index:
            break

        for delta in six_deltas:
            nx, ny, nz = idx_current[0] + delta[0], idx_current[1] + delta[1], idx_current[2] + delta[2]
            if 0 <= nx < length and 0 <= ny < width and 0 <= nz < height:
                if map[nx, ny, nz] == 0:
                    new_g = g_current + 1
                    if new_g < gv[nx, ny, nz]:
                        gv[nx, ny, nz] = new_g
                        parent_map[(nx, ny, nz)] = idx_current
                        h = heuristic((nx, ny, nz)) if astar else 0
                        heappush(open_list, (new_g + h, new_g, (nx, ny, nz)))

    if goal_index not in parent_map and idx_current != goal_index:
        return None, nodes_expanded

    n_current = goal_index
    while n_current is not None:
        idx.append(n_current)
        # path.append(occ_map.index_to_metric_negative_corner(n_current.index))
        n_current = parent_map.get(n_current)
    # Return a tuple (path, nodes_expanded)
    # path.reverse()
    idx.reverse()
    #path[0] = start
    #path[-1] = goal
    idx[0] = start_index
    idx[-1] = goal_index

    def ray_cast(ori):
        simp = [ori[0]]
        i_current = 0
        N = len(ori)
        while i_current < N - 1:
            i_next = N - 1
            while i_next > i_current:
                if no_collision(simp[-1], ori[i_next]):
                    simp.append(ori[i_next])
                    i_current = i_next
                    break
                i_next -= 1
            else:
                i_current += 1
                simp.append(ori[i_current])
        return simp


    #The below code is referenced from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing
    def bresenham3D(start, end):
        (x1, y1, z1), (x2, y2, z2) = start, end
        points = []
        dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1

        if dx >= dy and dx >= dz:
            p1, p2 = 2*dy - dx, 2*dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2*dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2*dx
                p1 += 2*dy
                p2 += 2*dz
                points.append((x1, y1, z1))
        elif dy >= dx and dy >= dz:
            p1, p2 = 2*dx - dy, 2*dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2*dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2*dy
                p1 += 2*dx
                p2 += 2*dz
                points.append((x1, y1, z1))
        else:
            p1, p2 = 2*dx - dz, 2*dy - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2*dz
                if p2 >= 0:
                    y1 += ys
                    p2 -= 2*dz
                p1 += 2*dx
                p2 += 2*dy
                points.append((x1, y1, z1))
        return points

    def no_collision(p1, p2):
        # mid_x, mid_y, mid_z = int(p1[0] + p2[0] / 2), int(p1[1] + p2[1] / 2), int(p1[2] + p2[2] / 2)
        for (x, y, z) in bresenham3D(p1, p2):
            if x < 0 or x >= length or y < 0 or y >= width or z < 0 or z >= height:
                return False
            if map[x, y, z] == 1:
                return False
        return True

    def rdp(points, epsilon):
        if len(points) < 3:
            return points
        start, end = points[0], points[-1]
        max_dist = 0
        index = 0

        for i in range(1, len(points) - 1):
            dist = point_line_distance(points[i], start, end)
            if dist > max_dist:
                max_dist = dist
                index = i

        if max_dist > epsilon:
            left = rdp(points[:index + 1], epsilon)
            right = rdp(points[index:], epsilon)
            return left[:-1] + right
        else:
            return [start, end]

    def point_line_distance(point, start, end):
        p = np.array(point)
        a = np.array(start)
        b = np.array(end)

        if np.array_equal(a, b):
            return np.linalg.norm(p - a)

        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab

        return np.linalg.norm(p - closest)

    def segment_insertion(points, max_length=8.0):
        if len(points) < 2:
            return points

        new_points = [points[0]]
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])
            vec = p2 - p1
            distance = np.linalg.norm(vec)

            num_segments = int(np.ceil(distance / max_length))
            if num_segments > 1:
                step = 1.0 / num_segments
                for j in range(1, num_segments):
                    new_point = p1 + (vec * j * step)
                    new_points.append(new_point.tolist())

            new_points.append(p2.tolist())

        new_points[0] = points[0]
        new_points[-1] = points[-1]
        return np.array(new_points)

    idx_ori = idx
    idx = ray_cast(idx_ori)
    # print(path)
    for i in idx:
        path.append(occ_map.index_to_metric_center(i))
    path_ori =[]
    for i in idx_ori:
        path_ori.append(occ_map.index_to_metric_center(i))
    path[0] = start
    path[-1] = goal
    path_ori[0] = start
    path_ori[-1] = goal
    epsilon = 0.2
    path = rdp(path, epsilon)
    path = segment_insertion(path)
    path = np.array(path)
    path_ori = np.array(path_ori)
    return path, path_ori
