from casadi import *
import math
import numpy as np

from DiffDriveModel import DiffDriveModel

class MPC:
    def __init__(self, T, N, max_v, max_w):
        self.T = T # horizon length
        self.N = N # discreate grid number
        self.dt = self.T/self.N # minute time
        self.nx = 3 # state variable number
        self.nu = 2 # input variable number
        self.nvar = self.nx + self.nu
        self.diffDrive = DiffDriveModel()

        self.Q  = [1.0, 1.0, 0.0]       # state weights[x, y, angle, v]
        self.Qf = [1.0, 1.0, 0.0]       # terminal state weights
        self.R  = [0.01, 0.01]          # input weights[v, w]
        self.V_W = 0.0

        max_v_input = max_v
        max_w_input = max_w

        w = [] # contain optimal variable
        w0 = [] # contain initial optimal variable
        lbw = [] # lower bound optimal variable
        ubw = [] # upper bound optimal variable
        J = 0 # cost function
        g  = [] # constrain
        lbg = [] # lower bound constrain
        ubg = [] # upper bound constrain
        lam_x0 = [] # Lagrangian multiplier
        lam_g0 = [] # Lagrangian multiplier

        Xk = MX.sym('X0', self.nx) # initial time state vector x0
        Pathref = MX.sym('path_ref', self.N, self.nx) # path reference

        w += [Xk]
        # equality constraint
        lbw += [0, 0, 0]  # constraints are set by setting lower-bound and upper-bound to the same value
        ubw += [0, 0, 0]      # constraints are set by setting lower-bound and upper-bound to the same value
        w0 +=  [0, 0, 0]      # x0 initial estimate
        lam_x0 += [0, 0, 0]    # Lagrangian multiplier initial estimate

        for k in range(self.N):
            Xref = MX.sym('x_ref', self.nx) # x reference
            Xref = Pathref[k,:]
            Uk = MX.sym('U_' + str(k), self.nu)
            w += [Uk]
            lbw += [-max_v_input, -max_w_input]
            ubw += [max_v_input, max_w_input]
            w0 += [0, 0]
            lam_x0 += [0, 0]

            # contouring error of path
            Ec = sin(Xref[2])*(Xk[0] - Xref[0]) - cos(Xref[2])*(Xk[1] - Xref[1])
            #lag path error of path
            El = -cos(Xref[2])*(Xk[0] - Xref[0]) - sin(Xref[2])*(Xk[1] - Xref[1])

            #stage cost
            J += self.stage_cost(Xk, Uk, Xref)
            J += (max_v_input - Uk[0])*self.V_W

            # Discretized equation of state by forward Euler
            dXk = self.diffDrive.dynamics(Xk, Uk)
            Xk_next = vertcat(Xk[0] + dXk[0] * self.dt,
                              Xk[1] + dXk[1] * self.dt,
                              Xk[2] + dXk[2] * self.dt)



            #arange angle
            #Xk_next[2] = atan2(sin(Xk_next[2]), cos(Xk_next[2]))

            Xk1 = MX.sym('X_' + str(k+1), self.nx)
            w   += [Xk1]
            lbw += [-inf, -inf, -inf]
            ubw += [inf, inf, inf]
            w0 += [0.0, 0.0, 0.0]
            lam_x0 += [0, 0, 0]

            # (xk+1=xk+fk*dt) is introduced as an equality constraint
            g   += [Xk_next-Xk1]
            lbg += [0, 0, 0]     # Equality constraints are set by setting lower-bound and upper-bound to the same value
            ubg += [0, 0, 0]     # Equality constraints are set by setting lower-bound and upper-bound to the same value
            lam_g0 += [0, 0, 0]
            Xk = Xk1

        # finite cost
        J += self.terminal_cost(Xk, Xref)

        self.J = J
        self.w = vertcat(*w)
        self.g = vertcat(*g)
        self.x = w0
        self.lam_x = lam_x0
        self.lam_g = lam_g0
        self.lbx = lbw
        self.ubx = ubw
        self.lbg = lbg
        self.ubg = ubg

        # NonLinearProblem
        self.nlp = {'f': self.J, 'x': self.w, 'p': Pathref, 'g': self.g}
        # Ipopt solver，min bariar parameter0.1，maximum iteration time 5, warm start ON
        self.solver = nlpsol('solver', 'ipopt', self.nlp, {'calc_lam_p':True, 'calc_lam_x':True, 'print_time':False, 'ipopt':{'max_iter':5, 'mu_min':0.1, 'warm_start_init_point':'yes', 'print_level':0, 'print_timing_statistics':'no'}})
        # self.solver = nlpsol('solver', 'scpgen', self.nlp, {'calc_lam_p':True, 'calc_lam_x':True, 'qpsol':'qpoases', 'print_time':False, 'print_header':False, 'max_iter':5, 'hessian_approximation':'gauss-newton', 'qpsol_options':{'print_out':False, 'printLevel':'none'}}) # print をオフにしたいがやり方がわからない

    def init(self, x0=None, path_ref=None):
        if x0 is not None:
            # set constraint for initial state
            self.lbx[0:self.nx] = x0
            self.ubx[0:self.nx] = x0
        # primal variables (x) dual variables solve（warm start）
        sol = self.solver(x0=self.x, p=path_ref, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        # save solution for next warm start
        self.x = sol['x'].full().flatten()
        self.lam_x = sol['lam_x'].full().flatten()
        self.lam_g = sol['lam_g'].full().flatten()

    def stage_cost(self, x, u, x_ref):
        cost = 0
        for i in range(self.nx):
            cost += 0.5 * self.Q[i] * (x[i] - x_ref[i])**2
        for i in range(self.nu):
            cost += 0.5 * self.R[i] * u[i]**2
        return cost

    def terminal_cost(self, x, x_ref):
        cost = 0
        for i in range(self.nx):
            cost += 0.5 * self.Q[i] * (x[i] - x_ref[i])**2
        return cost

    """
    x0 = np.array([x_current, y_current, v_current])
    xref = np.array([x_ref, y_ref, v_ref])
    """
    def solve(self, x0, path_ref):
        # set constraint for initial state
        self.lbx[0:self.nx] = x0
        self.ubx[0:self.nx] = x0
        # primal variables (x) dual variables solve（warm start）
        sol = self.solver(x0=self.x, p=path_ref, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        # save solution for next warm start
        self.x = sol['x'].full().flatten()
        self.lam_x = sol['lam_x'].full().flatten()
        self.lam_g = sol['lam_g'].full().flatten()

        return np.array([self.x[3], self.x[4]]) # return control input

    def get_path(self):
        path_x = []
        path_y = []

        #path_x.append(self.x[0])
        #path_y.append(self.x[1])
        for i in range(self.N):
            path_x.append(self.x[(self.nvar)*i])
            path_y.append(self.x[(self.nvar)*i+1])

        return path_x, path_y