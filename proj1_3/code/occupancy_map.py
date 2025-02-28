import heapq
import numpy as np
from scipy.spatial import Rectangle
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt

from flightsim.world import World
from flightsim import shapes
from pathlib import Path
import inspect
import json
class OccupancyMap:
    def __init__(self, world=World.empty((0, 2, 0, 2, 0, 2)), resolution=(.1, .1, .1), margin=.5):
        """
        This class creates a 3D voxel occupancy map of the configuration space from a flightsim World object.
        Parameters:
            world, a flightsim World object
            resolution, the discretization of the occupancy grid in x,y,z
            margin, the inflation radius used to create the configuration space (assuming a spherical drone)
        """
        self.world = world
        self.resolution = np.array(resolution)
        '''
        margin_lb = 0.0
        margin_ub = 0.6
        #self.margin = self.compute_dynamic_margin(world)
        #print(self.margin)
        self.margin = margin_lb
        self._init_map_from_world()

        density_lb = self.calculate_obstacle_density()
        self.margin = margin_ub
        self._init_map_from_world()
        density_ub = self.calculate_obstacle_density()
        inflation_scale = density_ub / density_lb
        self.compute_dynamic_margin(inflation_scale)
        '''
        self.margin = margin
        self._init_map_from_world()

    def calculate_obstacle_density(self):
        total_voxels = np.prod(self.map.shape)
        obstacle_voxels = np.sum(self.map)
        free_voxels = total_voxels - obstacle_voxels
        return obstacle_voxels / free_voxels if free_voxels > 0 else 1.0

    def compute_dynamic_margin(self, inflation_scale):
        print(inflation_scale)
        if inflation_scale > 14:
            self.margin = 0.35
            self._init_map_from_world()
        else:
            self.margin = 0.6
            self._init_map_from_world()

    def index_to_metric_negative_corner(self, index):
        """
        Return the metric position of the most negative corner of a voxel, given its index in the occupancy grid
        """
        return index*np.array(self.resolution) + self.origin

    def index_to_metric_center(self, index):
        """
        Return the metric position of the center of a voxel, given its index in the occupancy grid
        """
        return self.index_to_metric_negative_corner(index) + self.resolution/2.0

    def metric_to_index(self, metric):
        """
        Returns the index of the voxel containing a metric point.
        Remember that this and index_to_metric and not inverses of each other!
        If the metric point lies on a voxel boundary along some coordinate,
        the returned index is the lesser index.
        """
        return np.floor((metric - self.origin)/self.resolution).astype('int')

    def _metric_block_to_index_range(self, bounds, outer_bound=True):
        """
        A fast test that returns the closed index range intervals of voxels
        intercepting a rectangular bound. If outer_bound is true the returned
        index range is conservatively large, if outer_bound is false the index
        range is conservatively small.
        """

        # Implementation note: The original intended resolution may not be
        # exactly representable as a floating point number. For example, the
        # floating point value for "0.1" is actually bigger than 0.1. This can
        # cause surprising results on large maps. The solution used here is to
        # slightly inflate or deflate the resolution by the smallest
        # representative unit to achieve either an upper or lower bound result.
        sign = 1 if outer_bound else -1
        min_index_res = np.nextafter(self.resolution,  sign * np.inf) # Use for lower corner.
        max_index_res = np.nextafter(self.resolution, -sign * np.inf) # Use for upper corner.

        bounds = np.asarray(bounds)
        # Find minimum included index range.
        min_corner = bounds[0::2]
        min_frac_index = (min_corner - self.origin)/min_index_res
        min_index = np.floor(min_frac_index).astype('int')
        min_index[min_index == min_frac_index] -= 1
        min_index = np.maximum(0, min_index)
        # Find maximum included index range.
        max_corner = bounds[1::2]
        max_frac_index = (max_corner - self.origin)/max_index_res
        max_index = np.floor(max_frac_index).astype('int')
        max_index = np.minimum(max_index, np.asarray(self.map.shape)-1)
        return (min_index, max_index)


    def _init_map_from_world(self):
        """
        Creates the occupancy grid (self.map) as a boolean numpy array. True is
        occupied, False is unoccupied. This function is called during
        initialization of the object.
        """

        # Initialize the occupancy map, marking all free.
        bounds = self.world.world['bounds']['extents']
        voxel_dimensions_metric = []
        voxel_dimensions_indices = []
        for i in range(3):
            voxel_dimensions_metric.append(abs(bounds[1+i*2]-bounds[i*2]))
            voxel_dimensions_indices.append(int(np.ceil(voxel_dimensions_metric[i]/self.resolution[i])))
        self.map = np.zeros(voxel_dimensions_indices, dtype=bool)
        self.origin = np.array(bounds[0::2])

        # Iterate through each block obstacle.
        for block in self.world.world.get('blocks', []):
            extent = block['extents']
            outer_extent = extent + self.margin * np.array([-1, 1, -1, 1, -1, 1])
            (outer_min, outer_max) = self._metric_block_to_index_range(outer_extent)

            i, j, k = np.mgrid[outer_min[0]:outer_max[0] + 1,
                      outer_min[1]:outer_max[1] + 1,
                      outer_min[2]:outer_max[2] + 1]
            indices = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)

            voxel_min = self.origin + indices * self.resolution
            voxel_max = voxel_min + self.resolution

            dx = np.maximum(extent[0] - voxel_max[:, 0], voxel_min[:, 0] - extent[1])
            dy = np.maximum(extent[2] - voxel_max[:, 1], voxel_min[:, 1] - extent[3])
            dz = np.maximum(extent[4] - voxel_max[:, 2], voxel_min[:, 2] - extent[5])
            dist = np.sqrt(np.sum(np.where(np.array([dx, dy, dz]) > 0, np.array([dx, dy, dz]), 0) ** 2, axis=0))

            mask = (dist <= self.margin) & (~self.map[indices[:, 0], indices[:, 1], indices[:, 2]])
            self.map[indices[mask, 0], indices[mask, 1], indices[mask, 2]] = True

    def draw_filled(self, ax):
        """
        Visualize the occupancy grid (mostly for debugging)
        Warning: may be slow with O(10^3) occupied voxels or more
        Parameters:
            ax, an Axes3D object
        """
        self.world.draw_empty_world(ax)
        it = np.nditer(self.map, flags=['multi_index'])
        while not it.finished:
            if self.map[it.multi_index] == True:
                metric_loc = self.index_to_metric_negative_corner(it.multi_index)
                xmin, ymin, zmin = metric_loc
                xmax, ymax, zmax = metric_loc + self.resolution
                c = shapes.Cuboid(ax, xmax-xmin, ymax-ymin, zmax-zmin, alpha=0.1, linewidth=1, edgecolors='k', facecolors='b')
                c.transform(position=(xmin, ymin, zmin))
            it.iternext()

    def _draw_voxel_face(self, ax, index, direction):
        # Normalized coordinates of the top face.
        face = np.array([(1,1,1), (-1,1,1), (-1,-1,1), (1,-1,1)])
        # Rotate to find normalized coordinates of target face.
        if   direction[0] != 0:
            axis = np.array([0, 1, 0]) * np.pi/2 * direction[0]
        elif direction[1] != 0:
            axis = np.array([-1, 0, 0]) * np.pi/2 * direction[1]
        elif direction[2] != 0:
            axis = np.array([1, 0, 0]) * np.pi/2 * (1-direction[2])
        face = (Rotation.from_rotvec(axis).as_matrix() @ face.T).T
        # Scale, position, and draw using Face object.
        face = 0.5 * face * np.reshape(self.resolution, (1,3))
        f = shapes.Face(ax, face, alpha=0.1, linewidth=1, edgecolors='k', facecolors='b')
        f.transform(position=(self.index_to_metric_center(index)))

    def draw_shell(self, ax):
        self.world.draw_empty_world(ax)
        it = np.nditer(self.map, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            if self.map[idx] == True:
                for d in [(0,0,-1), (0,0,1), (0,-1,0), (0,1,0), (-1,0,0), (1,0,0)]:
                    neigh_idx = (idx[0]+d[0], idx[1]+d[1], idx[2]+d[2])
                    neigh_exists = self.is_valid_index(neigh_idx)
                    if not neigh_exists or (neigh_exists and not self.map[neigh_idx]):
                        self._draw_voxel_face(ax, idx, d)
            it.iternext()

    def draw(self, ax):
        self.draw_shell(ax)

    def is_valid_index(self, voxel_index):
        """
        Test if a voxel index is within the map.
        Returns True if it is inside the map, False otherwise.
        """
        for i in range(3):
            if voxel_index[i] >= self.map.shape[i] or voxel_index[i] < 0:
                return False
        return True

    def is_valid_metric(self, metric):
        """
        Test if a metric point is within the world.
        Returns True if it is inside the world, False otherwise.
        """
        bounds = self.world.world['bounds']['extents']
        for i in range(3):
            if metric[i] <= bounds[i*2] or metric[i] >= bounds[i*2+1]:
                return False
        return True

    def is_occupied_index(self, voxel_index):
        """
        Test if a voxel index is occupied.
        Returns True if occupied or outside the map, False otherwise.
        """
        return (not self.is_valid_index(voxel_index)) or self.map[tuple(voxel_index)]

    def is_occupied_metric(self, voxel_metric):
        """
        Test if a metric point is within an occupied voxel.
        Returns True if occupied or outside the map, False otherwise.
        """
        ind = self.metric_to_index(voxel_metric)
        return (not self.is_valid_index(ind)) or self.is_occupied_index(ind)