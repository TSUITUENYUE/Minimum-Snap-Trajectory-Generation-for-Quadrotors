import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """
        # STUDENT CODE HERE
        self.v = 2.5
        self.waypoint = points
        self.N = len(points)
        self.dist = np.linalg.norm(points[1:] - points[:-1],axis=1)
        self.tf = np.sum(self.dist) / self.v
        self.direction = (points[1:] - points[:-1]) / self.dist[:,None]
        self.vel = self.v * self.direction
        self.duration = self.dist/self.v

        self.start_time = np.concatenate([[0],np.cumsum(self.duration[:-1])])
        self.transition_time = self.duration / 4



    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        if t<= self.tf:
            for i in range(self.N-1):
                if t>=self.start_time[i] and t<self.start_time[i]+self.duration[i]:
                    x = self.waypoint[i] + self.vel[i] * (t - self.start_time[i])
                    x_dot = self.vel[i]
                    x_ddot = np.zeros((3,))
                    x_dddot = np.zeros((3,))
                    x_ddddot = np.zeros((3,))
                    yaw = 0
                    yaw_dot = 0
                    break
        else:
            x = self.waypoint[-1]
            x_dot = np.zeros((3,))
            x_ddot = np.zeros((3,))
            x_dddot = np.zeros((3,))
            x_ddddot = np.zeros((3,))
            yaw = 0
            yaw_dot = 0

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output

