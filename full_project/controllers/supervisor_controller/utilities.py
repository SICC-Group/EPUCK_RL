import math

import numpy as np


def normalize_to_range(value, min, max, new_min, new_max):
    min = float(min)
    max = float(max)
    new_min = float(new_min)
    new_max = float(new_max)
    return (new_max - new_min) / (max - min) * (value - max) + new_max


def get_distance_from_target(robot_node, target_node):
    distance_from_target = np.linalg.norm(robot_node[:, np.newaxis, :] - target_node, axis=2)
    return distance_from_target


def get_angle_from_target(robot_node,
                          target_node,
                          epuck_angle,
                          is_abs=False):
    """
    Returns the angle between the facing vector of the robot and the target position.
    Explanation can be found here https://math.stackexchange.com/a/14180.
    :param robot_node: The robot Webots node
    :type robot_node: controller.node.Node
    :param target_node: The target Webots node
    :type target_node: controller.node.Node
    :param is_abs: Whether to return the absolute value of the angle.
    :type is_abs: bool
    :return: The angle between the facing vector of the robot and the target position
    :rtype: float, [-π, π]
    """
    # The sign of the z-axis is needed to flip the rotation sign, because Webots seems to randomly
    # switch between positive and negative z-axis as the robot rotates.
    angle_between = np.arctan2(target_node[:, np.newaxis, 1] - robot_node[:, 1], target_node[:, np.newaxis, 0] - robot_node[:, 0])
    angle_diff = np.fmod(angle_between.T - epuck_angle,math.tau)

    return abs(angle_diff) if is_abs else angle_diff
