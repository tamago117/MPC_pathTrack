import numpy as np
import math
import matplotlib.pyplot as plt

from path_generate import make_track, make_side_lane
from MPC import MPC
from DiffDriveModel import DiffDriveModel

sim_time = 50.0
sampling_time = 0.05 # 100hz
sim_steps = math.floor(sim_time / sampling_time)

NX = 3  # x = x, y, v, yaw
T = 20  # horizon length

MAX_V = 5
MAX_W = 5
DL = MAX_V * sampling_time

TRACK_LENGTH = 10
TRACK_RADIUS = 5

N_IND_SEARCH = 10  # Search index number

def calc_nearest_index(state, cx, cy, cind):

    dx = [state[0] - icx for icx in cx[cind:(cind + N_IND_SEARCH)]]
    dy = [state[1] - icy for icy in cy[cind:(cind + N_IND_SEARCH)]]

    distanceList = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    min_distance = min(distanceList)

    nearest_index = distanceList.index(min_distance) + cind

    return nearest_index

def calc_ref_trajectory(state, cx, cy, cyaw, cind):
    xref = np.zeros((T, NX))
    ncourse = len(cx)

    ind = calc_nearest_index(state, cx, cy, cind)

    if cind >= ind:
        ind = cind

    xref[0, 0] = cx[ind]
    xref[0, 1] = cy[ind]
    xref[0, 2] = cyaw[ind]

    travel = 0.0

    for i in range(T):
        #travel += abs(state.v) * DT
        travel += MAX_V * sampling_time
        dind = int(round(travel / DL))

        if (ind + dind) < ncourse:
            xref[i, 0] = cx[ind + dind]
            xref[i, 1] = cy[ind + dind]
            xref[i, 2] = cyaw[ind + dind]
        else:
            xref[i, 0] = cx[ncourse - 1]
            xref[i, 1] = cy[ncourse - 1]
            xref[i, 2] = cyaw[ncourse - 1]

    return xref, ind

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, robot_radius):  # pragma: no cover
    circle = plt.Circle((x, y), robot_radius, color="b")
    plt.gcf().gca().add_artist(circle)
    out_x, out_y = (np.array([x, y]) +
                    np.array([np.cos(yaw), np.sin(yaw)]) * robot_radius)
    plt.plot([x, out_x], [y, out_y], "-k")

def main():
    x = np.array([0.0, 0.0, 0.0])
    track_path = make_track(circle_radius=TRACK_RADIUS, linelength=TRACK_LENGTH, dl = DL)
    right_lane, left_lane = make_side_lane(track_path, lane_width=1.0)

    diffDrive = DiffDriveModel()
    mpc = MPC(sampling_time*T, T, MAX_V, MAX_W)

    xs = []

    current_index = 0
    for step in range(sim_steps):
        if step%(1/sampling_time) == 0:
            print('t=', step*sampling_time)

        reference_path, current_index = calc_ref_trajectory(x, track_path[:, 0], track_path[:, 1], track_path[:, 2], current_index)

        u = mpc.solve(x, reference_path)
        estimated_path = mpc.get_path()

        # data store
        xs.append(x)
        xs1 = [x[0] for x in xs]
        xs2 = [x[1] for x in xs]
        x = x + sampling_time * np.array(diffDrive.dynamics(x, u))

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        # track
        plt.plot(track_path[:, 0], track_path[:, 1])
        plt.plot(right_lane[:, 0], right_lane[:, 1])
        plt.plot(left_lane[:, 0], left_lane[:, 1])
        # robot
        plot_robot(x[0], x[1], x[2], 0.3)
        plot_arrow(x[0], x[1], x[2])
        # estimated path
        plt.plot(estimated_path[0], estimated_path[1])
        # reference_path
        plt.scatter(reference_path[:, 0], reference_path[:, 1])
        # runned trajectory
        plt.plot(xs1, xs2)

        plt.title(f"MPC path track\n v: {u[0]:.2f} , w: {u[1]:.2f}")

        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if step*sampling_time>1.0:
            if math.sqrt((track_path[-1,0]-x[0])**2+(track_path[-1, 1]-x[1])**2)<0.05:
                break

if __name__ == '__main__':
    main()