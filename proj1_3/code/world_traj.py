import numpy as np
from numpy.ma.core import equal
from flightsim.shapes import Quadrotor
from .graph_search import graph_search
from cvxopt import matrix, solvers
from scipy.linalg import block_diag
from scipy.special import factorial
class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        #self.resolution = np.array([0.25, 0.25, 0.25])
        self.resolution = np.ones([3]) * 0.085
        self.margin = 0.6
        self.bounds = world.world['bounds']['extents']
        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, path_ori = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        # self.points = np.zeros((1,3)) # shape=(n_pts,3)
        self.points = self.path # since my path is already simplified in graph_search using ray casting, I directly use it as my waypoints.
        # print(self.points)
        self.path = path_ori
        # print(self.points.shape)
        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        self.n_order = 7
        self.d_r = 4
        self.d_phi = 2
        self.n_coefficients = self.n_order + 1
        self.n_frac = 4

        self.t_start = np.array([])
        self.t_duration = np.array([])
        self.n_segments = self.points.shape[0] - 1

        self.a = np.zeros((3, self.n_segments, self.n_coefficients)) # 3 * 8 * n_segments, lower order at front

        self.v_max = 2.75
        #self.a_max = 1.0
        self.dist = np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)

        self.t_max = np.sum(self.dist) / self.v_max

        self.compute_K_matrix()
        self.cost = 0
        self.use_ADAM = False
        self.use_corridor = False
        self.optimize_time_segment(use_grad=True)
        self.generate_trajectory()


    def _update_time(self):
        self.t_start = np.concatenate([np.zeros(1), np.cumsum(self.t_duration)])
        self.t_max = np.sum(self.t_duration)

    def time_constraints(self):
        self.t_duration = np.clip(self.t_duration, 0.1, None)
        self.t_max = np.sum(self.t_duration)
        #total_time = np.sum(self.t_duration)
        #self.t_duration = self.t_duration * (self.t_max / total_time)
        self._update_time()


    def optimize_time_segment(self, use_grad=True):
        # weights = self.dist / np.sum(self.dist)
        weights = np.sqrt(self.dist) / np.sum(np.sqrt(self.dist))
        self.t_duration = weights * self.t_max

        self._update_time()
        if use_grad:
            max_iter = 200
            min_iter = 50
            lr_0 = 2e-2 * self.n_segments
            lr = lr_0
            lr_min = 1e-3
            h = 1e-6
            threshold = 1e-1
            threshold2 = 1e-3
            # print(self.n_segments)
            g = np.ones((self.n_segments, self.n_segments)) * (-1 / (self.n_segments - 1))
            np.fill_diagonal(g, 1)
            grad = np.zeros(self.n_segments)


            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            m = np.zeros(self.n_segments)
            v = np.zeros(self.n_segments)

            grad_ema = np.zeros(self.n_segments)
            grad_var = np.ones(self.n_segments)
            beta_ema = 0.9
            epsilon2 = 1e-8

            loss_prev = np.inf
            for t in range(1, max_iter + 1):
                if self.generate_trajectory():
                    cost = self.cost
                    t_original = self.t_duration.copy()
                    for i in range(self.n_segments):
                        self.t_duration = t_original + h * g[i]
                        self.time_constraints()
                        self.generate_trajectory(use_for_grad=use_grad)
                        cost_i = self.cost
                        grad[i] = (cost_i - cost) / (h * self.n_segments * self.n_segments)

                    '''
                    grad_ema = beta_ema * grad_ema + (1 - beta_ema) * grad
                    grad_var = beta_ema * grad_var + (1 - beta_ema) * grad ** 2

                    scale_factor = 1.0 / (np.sqrt(grad_var) + epsilon2)

                    max_scale = 1.0 / (np.median(np.abs(grad)) + epsilon2)
                    scale_factor = np.clip(scale_factor, None, max_scale)

                    scaled_grad = grad * scale_factor

                    global_norm = np.linalg.norm(scaled_grad)
                    target_norm = np.sqrt(self.n_segments)
                    grad_normalized = scaled_grad * (target_norm / (global_norm + epsilon2))
                    #print(grad_normalized)
                    '''
                    grad_norm = np.linalg.norm(grad)
                    grad_normalized = grad / grad_norm

                    if self.use_ADAM:
                        m = beta1 * m + (1 - beta1) * grad_normalized
                        v = beta2 * v + (1 - beta2) * (grad_normalized ** 2)
                        m_hat = m / (1 - beta1 ** t)
                        v_hat = v / (1 - beta2 ** t)

                        t_update = lr * m_hat / (np.sqrt(v_hat) + epsilon)
                        #t_update_2 = lr * grad_normalized
                        print(np.linalg.norm(m_hat / (np.sqrt(v_hat) + epsilon)))
                    else:
                        t_update = lr * grad_normalized
                    #print(np.linalg.norm(grad))
                    self.t_duration = t_original - t_update
                    self.time_constraints()
                    # print(t_original, t_update)
                    # print(lr)
                    grad_norm = np.linalg.norm(grad)
                    loss = grad_norm / self.n_segments
                    d_loss = abs(loss_prev - loss)
                    loss_prev = loss

                    lr = max(lr * (1 - (t / 75) ** 3),lr_min)
                    #lr = lr * (1 - (t / max_iter))
                    # lr = lr * min(1, np.sqrt(grad_norm / norm_prev))
                    print("iter {}: loss = {}: lr = {}: norm = {}".format(t, loss, lr, grad_norm))
                    if loss < threshold and t > min_iter:
                        break
                    if d_loss < threshold2 and t > min_iter:
                        break
                else:
                    self.t_duration = t_original
                    #max_iter *= 2
                    #lr = lr_0 / 2
                    break
                self.time_constraints()
                #print(self.t_duration)
                #print(self.t_max)


    def compute_Q(self):
        Q_list = []
        k = np.arange(self.n_coefficients)

        alpha = np.where(k >= self.d_r, k * (k - 1) * (k - 2) * (k - 3), 0)
        u, v = np.meshgrid(k, k)
        mask = (u >= self.d_r) & (v >= self.d_r)
        beta = np.outer(alpha, alpha)
        denom = u + v - 7
        denom = np.where(denom == 0, 1, denom)
        # print(denom)

        for i in range(self.n_segments):
            dt = self.t_duration[i]
            Qi = np.zeros((self.n_coefficients, self.n_coefficients)) # shape: n_coefficients ** 2
            Qi[mask] = beta[mask] * (dt ** (denom[mask])) / (denom[mask])
            Q_list.append(Qi)

        return Q_list

    def compute_K_matrix(self):
        self.K = np.zeros([self.d_r, self.n_coefficients])
        k = np.arange(self.n_coefficients)
        self.K[0,:] = 1.0
        for i in range(1, self.d_r):
            self.K[i,i:] = factorial(k[i:]) / factorial(k[i:] - i)


    def compute_derivate_matrix(self, dt):
        D = np.zeros([self.d_r, self.n_coefficients])
        # print(self.K)
        D[0,:] = np.power(dt, np.arange(self.n_coefficients))
        for i in range(1,self.d_r):
            exp = np.arange(self.n_coefficients-i)
            # print(np.power(dt, exp) , self.K[i])
            D[i,i:] = np.power(dt, exp) * self.K[i,i:]
        # print(D)
        return D

    def compute_equality_constraints(self):
        n_constraints = (2 * self.n_segments)  * self.d_r
        # print(n_constraints)
        A = np.zeros([3, n_constraints,self.n_segments * self.n_coefficients])
        b = np.zeros([3, n_constraints])
        #print(self.n_segments)
        #print(A.shape)
        # print(self.points.shape)
        #print(b.shape)
        pos_slice = slice(0, n_constraints, 2 * self.d_r)
        #print(b[:,pos_slice].shape)
        b[:,pos_slice] = self.points[:-1].T
        b[:,-self.d_r] = self.points[-1].T
        D0 = self.compute_derivate_matrix(0)
        D0 = np.broadcast_to(D0,[3, self.d_r, self.n_coefficients])

        A[:,:self.d_r,:self.n_coefficients] = D0
        Dim1 = self.compute_derivate_matrix(self.t_duration[0])
        for i in range(1, self.n_segments):
            dt = self.t_duration[i]
            Di = self.compute_derivate_matrix(0)
            # print(Di, Dim1)
            Di = np.broadcast_to(Di,[3, self.d_r, self.n_coefficients])
            Dim1 = np.broadcast_to(Dim1,[3, self.d_r, self.n_coefficients])

            prev_slice = slice((i - 1) * self.n_coefficients,i * self.n_coefficients)
            current_slice = slice(i * self.n_coefficients,(i + 1) * self.n_coefficients)

            prev_row = slice((2 * i - 1) * self.d_r,2 * i * self.d_r)
            # current_row = slice((i+1) * self.d_r,(i+2) * self.d_r)
            current_row = 2 * i * self.d_r
            #continous constraints
            A[:, prev_row, prev_slice] = Dim1
            A[:, prev_row, current_slice] = -Di
            #pos constraints
            # A[:,current_row, current_slice] = Di
            A[:, current_row, current_slice] = Di[:,0,:]


            Dim1 = self.compute_derivate_matrix(dt)

        Df = self.compute_derivate_matrix(self.t_duration[-1])
        Df = np.broadcast_to(Df,[3, self.d_r, self.n_coefficients])

        A[:,-self.d_r:,-self.n_coefficients:] = Df

        # constraints = [A[i] @ coefficients[i] == b[i] for i in range(3)]
        # print(A[0])
        #with open("A.txt", "w") as f:
        #    np.savetxt(f, A[0], fmt="%.6f")
        #with open("b.txt", "w") as f:
        #   np.savetxt(f, b[0], fmt="%.6f")
        # print(np.linalg.matrix_rank(A[0]) * 3)
        return A, b

    def compute_intermidiate_matrix(self, dt):
        seg_fractions = (np.arange(1, self.n_frac + 1) / (self.n_frac + 1))
        t_fractions = dt * seg_fractions
        F = np.vander(t_fractions, self.n_coefficients, increasing=True)
        return F,seg_fractions

    def compute_corridor_length(self, center):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        dx = min(center[0] - xmin, xmax - center[0]) - 0.1
        dy = min(center[1] - ymin, ymax - center[1]) - 0.1
        dz = min(center[2] - zmin, zmax - center[2]) - 0.1
        #safe_radius = min(dx, dy, dz)
        #safe_radius = max(safe_radius, 0.1)

        return np.array([dx,dy,dz])

    def compute_inequality_constraints(self):
        n_constraints = self.n_segments * self.n_frac * 2
        A = np.zeros([3, n_constraints, self.n_segments * self.n_coefficients])
        b = np.zeros([3, n_constraints])
        vec_t = self.points[1:] - self.points[:-1]

        for i in range(self.n_segments):
            current_slice = slice(i * self.n_coefficients, (i + 1) * self.n_coefficients)
            current_row_pos = slice(2 * i * self.n_frac, (2 * i + 1) * self.n_frac)
            current_row_neg = slice((2 * i + 1) * self.n_frac, 2 * (i + 1) * self.n_frac)

            F, seg_fractions = self.compute_intermidiate_matrix(self.t_duration[i])
            F = np.broadcast_to(F, [3, self.n_frac, self.n_coefficients])

            A[:, current_row_pos, current_slice] = F
            A[:, current_row_neg, current_slice] = -F

            center_points = self.points[i] + seg_fractions[:, None] * vec_t[i]

            for j in range(self.n_frac):
                current_center = center_points[j]
                corridor_width = self.compute_corridor_length(current_center)
                upper_bound = current_center + corridor_width
                lower_bound = current_center - corridor_width
                b[:, current_row_pos.start + j] = upper_bound
                b[:, current_row_neg.start + j] = -lower_bound

        return A, b


    def generate_trajectory(self,use_for_grad=False):
        Q_list = self.compute_Q()
        H = block_diag(*Q_list)
        # print(H)
        # print(3 * self.n_segments * self.n_coefficients , 3 * np.linalg.matrix_rank(H)) # rank = 24, required 48
        # print(H)# shape: (n_segments * n_coefficients) ** 2
        #H = np.broadcast_to(H,[3, self.n_segments * self.n_coefficients, self.n_segments * self.n_coefficients]) #shape: 3 * (n_segments * n_coefficients) ** 2 (3 dim)
        P = block_diag(H, H, H)
        q = np.zeros((3 *self.n_segments * self.n_coefficients,))
        #print(q.shape)
        A,b = self.compute_equality_constraints()
        G,h = self.compute_inequality_constraints()
        #print(np.linalg.matrix_rank(P))
        P = matrix(P)
        q = matrix(q)

        A = block_diag(A[0],A[1],A[2])
        #print(np.linalg.matrix_rank(A))
        mask = np.linalg.norm(A, axis=1) > 0
        A = A[mask, :]
        #print(np.linalg.matrix_rank(A))
        A = matrix(A)
        b = np.hstack([b[0],b[1],b[2]])
        b = b[mask]
        #print(b.shape)
        b = matrix(b)
        if self.use_corridor:
            G = block_diag(G[0],G[1],G[2])
            #print(np.linalg.matrix_rank(G))
            G = matrix(G)
            h = np.hstack([h[0],h[1],h[2]])
            #print(h.shape)
            h = matrix(h)
        else:
            G = None
            h = None
        solvers.options['show_progress'] = False
        try:
            result = solvers.qp(P, q, G, h, A, b)

            x = np.array((result['x'])).flatten()
            self.cost = result['primal objective']
            if not use_for_grad:
                self.a = x.reshape(3, self.n_segments, self.n_coefficients)
            return True
        except Exception:
            return False
        #coefficients = cp.Variable((3, self.n_segments * self.n_coefficients))
        #min_obj = cp.sum([cp.quad_form(coefficients[i, :], cp.psd_wrap(H[i])) for i in range(3)])

        #equal_constraints = self.compute_equality_constraints(coefficients)
        #inequal_constraints = self.compute_inequality_constraints(coefficients)

        # prob = cp.Problem(cp.Minimize(min_obj), equal_constraints)
        #prob = cp.Problem(cp.Minimize(min_obj), [*equal_constraints, *inequal_constraints])
        # print(prob)
        #result = prob.solve(solver=cp.ECOS)
        #result = prob.solve(solver=cp.OSQP)
        #self.cost = result
        #print(result)
        # print(coefficients.value)
        #if not use_for_grad:
        #    self.a = coefficients.value.reshape(3 , self.n_segments, self.n_coefficients)


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
        if t<=self.t_max:
            idx = np.searchsorted(self.t_start[1:], t)
            dt = t - self.t_start[idx]
            # k = np.arange(self.n_coefficients)
            #print(t, idx, self.t_start[idx], dt)
            dt_powers = np.array([dt ** i for i in range(self.n_coefficients)])
            x = (self.a[:, idx, :] * dt_powers).sum(axis=1)

            x_dot_coefficients = self.a[:, idx, 1:] * np.arange(1, self.n_coefficients)
            x_dot = (x_dot_coefficients * dt_powers[:-1]).sum(axis=1)
            #vx = x_dot[0]
            #vy = x_dot[1]
            #if np.linalg.norm(x_dot[0:2]) > 0.1:
            #    yaw = np.arctan2(vy, vx)

        else:
            x = self.points[-1]

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
