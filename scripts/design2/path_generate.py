import numpy as np
import math
import matplotlib.pyplot as plt

def circle(center_x, center_y, radius, start=0., end=2*np.pi, dl=0.1):
    """ Create circle matrix
    Args:
        center_x (float): the center x position of the circle
        center_y (float): the center y position of the circle
        radius (float): in meters
        start (float): start angle
        end (float): end angle
    Returns:
        circle x : numpy.ndarray
        circle y : numpy.ndarray
    """

    diff = end - start

    arc_length = 2*np.pi*radius*(diff/(2*np.pi))
    n_point = round(arc_length/dl)

    circle_xs = []
    circle_ys = []

    for i in range(n_point + 1):
        circle_xs.append(center_x + radius * np.cos(i*diff/n_point + start))
        circle_ys.append(center_y + radius * np.sin(i*diff/n_point + start))

    return np.array(circle_xs), np.array(circle_ys)

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def make_track(circle_radius, linelength, dl):
    """ make track
    Input parameters:
        circle_radius (float): circle radius
        linelength (float): line length

    Returns:
        road (numpy.ndarray): shape(n_point, 3) x, y, angle
    """
    line_points = round(linelength/dl)

    line = np.linspace(-linelength/2, linelength/2, num=line_points+1, endpoint=False)[1:]
    line_1 = np.stack((line, np.zeros(line_points)), axis=1)
    line_2 = np.stack((line[::-1], np.zeros(line_points)+circle_radius*2.), axis=1)
    line_3 = np.stack((line, np.zeros(line_points)), axis=1)

    # circle
    circle_1_x, circle_1_y = circle(linelength/2., circle_radius,
                                    circle_radius, start=-np.pi/2., end=np.pi/2., dl=dl)
    circle_1 = np.stack((circle_1_x, circle_1_y), axis=1)

    circle_2_x, circle_2_y = circle(-linelength/2., circle_radius,
                                    circle_radius, start=np.pi/2., end=3*np.pi/2., dl=dl)
    circle_2 = np.stack((circle_2_x, circle_2_y), axis=1)

    road_pos = np.concatenate((line_1, circle_1, line_2, circle_2, line_3), axis=0)

    # calc road angle
    road_diff = road_pos[1:] - road_pos[:-1]
    road_angle = np.arctan2(road_diff[:, 1], road_diff[:, 0])
    road_angle = np.concatenate((np.zeros(1), road_angle))

    road = np.concatenate((road_pos, road_angle[:, np.newaxis]), axis=1)

    road[:, 0] = road[:, 0] + linelength/2

    return road

def make_side_lane(road, lane_width):
    """make_side_lane
    Input parameters:
        road (numpy.ndarray): shape(n_point, 3) x, y, angle
        lane_width (float): width of the lane
    Output:
        right_lane (numpy.ndarray): shape(n_point, 3) x, y, angle
        left_lane  (numpy.ndarray): shape(n_point, 3) x, y, angle
    """
    right_lane_x = lane_width/2*np.cos(road[:,2]-np.pi/2) +road[:,0]
    right_lane_y = lane_width/2*np.sin(road[:,2]-np.pi/2) +road[:,1]
    right_lane_pos = np.stack((right_lane_x, right_lane_y), axis=1)

    left_lane_x = lane_width/2*np.cos(road[:,2]+np.pi/2) +road[:,0]
    left_lane_y = lane_width/2*np.sin(road[:,2]+np.pi/2) +road[:,1]
    left_lane_pos = np.stack((left_lane_x, left_lane_y), axis=1)

    road_angle = road[:,2]

    right_lane = np.concatenate((right_lane_pos, road_angle[:, np.newaxis]), axis=1)
    left_lane = np.concatenate((left_lane_pos, road_angle[:, np.newaxis]), axis=1)

    return right_lane, left_lane

if __name__ == '__main__':
    road = make_track(circle_radius = 1, linelength = 2, dl=0.1)
    right_lane, left_lane = make_side_lane(road, lane_width=0.5)
    print(road)
    # road
    plt.plot(road[:, 0], road[:, 1])
    plt.plot(right_lane[:, 0], right_lane[:, 1])
    plt.plot(left_lane[:, 0], left_lane[:, 1])
    # arrow plot
    for i in range(int(road[:,0].size/10)):
        plot_arrow(road[i*10, 0], road[i*10, 1], road[i*10, 2], 0.1, 0.05)

    plt.show()
