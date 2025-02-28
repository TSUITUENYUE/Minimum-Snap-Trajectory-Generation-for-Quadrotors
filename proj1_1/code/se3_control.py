import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE

        # v = 2.5

        self.kp = np.array([22.3,22.3,22.3])
        self.kd = np.array([15.5,15.5,15.5])
        #self.kp = np.array([2,2,5])
        # self.kd = np.array([6,6,10])

        self.kd_phi = 12
        self.kp_phi = 10

        self.kd_cita = 12
        self.kp_cita = 10

        self.kd_yaw = 12
        self.kp_yaw = 5

        self.KR = np.diag([2277,2277,2277])
        self.Kw = np.diag([177, 177, 177])
        #self.KR = np.diag([2000,2000,100])
        #self.Kw = np.diag([150, 150, 80])

        self.flag = "nonlinear"



    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        u1 = cmd_thrust
        u2 = cmd_moment
        R_des = np.zeros((3,3))

        x = state['x']
        v = state['v']
        q = state['q']
        w = state['w']

        i,j,k,omega = q

        sin_yaw = 2 * (omega * k + i * j)
        cos_yaw = 1 - 2 * (j ** 2 + k ** 2)
        yaw = np.arctan2(sin_yaw, cos_yaw)

        sin_roll = 2 * (omega * i - j * k)
        sin_roll = np.clip(sin_roll, -1.0, 1.0)
        phi = np.arcsin(sin_roll)

        sin_pitch = 2 * (omega * j + i * k)
        cos_pitch = 1 - 2 * (i ** 2 + k ** 2)
        cita = np.arctan2(sin_pitch, cos_pitch)


        def quaternion_to_R(yaw,phi,cita):
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0],
                           [0, 0, 1]])
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(phi), -np.sin(phi)],
                           [0, np.sin(phi), np.cos(phi)]])
            Ry = np.array([[np.cos(cita), 0, np.sin(cita)],
                           [0, 1, 0],
                           [-np.sin(cita), 0, np.cos(cita)]])
            R = Rz @ Rx @ Ry
            return R

        def R_to_quaternion(R):
            w = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
            if w > 1e-6:
                i = (R[2, 1] - R[1, 2]) / (4 * w)
                j = (R[0, 2] - R[2, 0]) / (4 * w)
                k = (R[1, 0] - R[0, 1]) / (4 * w)
            else:
                trace = np.trace(R)
                if trace > 0:
                    S = np.sqrt(trace + 1.0) * 2
                    w = 0.25 * S
                    i = (R[2, 1] - R[1, 2]) / S
                    j = (R[0, 2] - R[2, 0]) / S
                    k = (R[1, 0] - R[0, 1]) / S
                elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                    w = (R[2, 1] - R[1, 2]) / S
                    i = 0.25 * S
                    j = (R[0, 1] + R[1, 0]) / S
                    k = (R[0, 2] + R[2, 0]) / S
                elif R[1, 1] > R[2, 2]:
                    S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                    w = (R[0, 2] - R[2, 0]) / S
                    i = (R[0, 1] + R[1, 0]) / S
                    j = 0.25 * S
                    k = (R[1, 2] + R[2, 1]) / S
                else:
                    S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                    w = (R[1, 0] - R[0, 1]) / S
                    i = (R[0, 2] + R[2, 0]) / S
                    j = (R[1, 2] + R[2, 1]) / S
                    k = 0.25 * S
            q = np.array([i, j, k, w])
            q /= np.linalg.norm(q)
            return q
        # print(R_to_quaternion(quaternion_to_R(0,np.radians(45),0)))
        R = quaternion_to_R(yaw,phi,cita)

        p = w[0]
        q = w[1]
        r = w[2]

        x_t = flat_output['x']
        v_t = flat_output['x_dot']
        a_t = flat_output['x_ddot']
        yaw_t = flat_output['yaw']
        yaw_dot_t = flat_output['yaw_dot']

        w_des = np.zeros((3,))
        # w_des =

        a_des = a_t - self.kd * (v - v_t) - self.kp * (x - x_t)

        if self.flag == "linear":
            u1 = (a_des[2] + self.g) * self.mass
            A = self.g * np.array([[np.cos(yaw_t),np.sin(yaw_t)],
                                   [np.sin(yaw_t),-np.cos(yaw_t)]])
            b = np.array([a_des[0], a_des[1]])
            angle_des = np.linalg.inv(A) @ b
            cita_des = angle_des[0]
            phi_des = angle_des[1]
            p_des = q_des = 0
            R_des = quaternion_to_R(yaw_t,phi_des,cita_des)
            u2 = self.inertia @ np.array([-self.kp_phi * (phi- phi_des) - self.kd_phi * (p - p_des),
                                          -self.kp_cita * (cita- cita_des) - self.kd_cita * (q - q_des),
                                          -self.kp_yaw * (yaw - yaw_t) - self.kd_yaw * (r - yaw_dot_t)])

        elif self.flag == "nonlinear":
            F_des = self.mass * (a_des + np.array([0,0,self.g]))

            F_norm = np.linalg.norm(F_des)
            if F_norm < 1e-6:
                b3_des = np.array([0, 0, 1])
                F_des = np.array([0, 0, self.mass * self.g])
            else:
                b3_des = F_des / F_norm

            b3 = R @ np.array([0,0,1]).T
            u1 = b3.T @ F_des

            a_yaw = np.array([np.cos(yaw_t),np.sin(yaw_t),0])
            b2_des = np.cross(b3_des,a_yaw)
            b2_des_norm = np.linalg.norm(b2_des)

            if b2_des_norm < 1e-6:
                b2_des = np.array([-np.sin(yaw_t), np.cos(yaw_t), 0])
            else:
                b2_des = b2_des / b2_des_norm

            b1_des = np.cross(b2_des,b3_des)
            b1_des /= np.linalg.norm(b1_des)

            R_des = np.array([b1_des,b2_des,b3_des]).T
            R_err = 0.5 * (R_des.T @ R - R.T @ R_des)

            e_R = np.array([R_err[2,1],R_err[0,2],R_err[1,0]])
            e_w = w - w_des

            u2 = self.inertia @ (-self.KR @ e_R - self.Kw @ e_w)


        #print("t:",t, "eR",e_R, "ew",e_w)
        gamma = self.k_drag / self.k_thrust
        L = self.arm_length
        M = np.array([
            [1, 1, 1, 1],
            [0, L, 0, -L],
            [-L, 0, L, 0],
            [gamma, -gamma, gamma, -gamma]
        ])

        B = np.array([u1, u2[0], u2[1], u2[2]])

        F = np.clip(np.linalg.solve(M, B),0,None)
        speed_sqr = F / self.k_thrust
        # print("t",t, "F", F,"a",a_des,"v",v_t)
        cmd_motor_speeds = np.clip(np.sqrt(speed_sqr),self.rotor_speed_min,self.rotor_speed_max)
        cmd_thrust = np.clip(u1,0,None)
        cmd_moment = u2
        cmd_q = R_to_quaternion(R_des)
        #print(cmd_motor_speeds)
        #print(cmd_thrust)
        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
