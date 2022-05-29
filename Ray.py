from matplotlib import pyplot as plt
import numpy as np


class Ray:

    def __init__(self, antenna_img, point_rx):
        self.c = 3 * 10 ** 8  # speed of light
        self.gamma = 1  # coeff reflexion
        self.D = 1  # coeff diffraction
        self.d = np.linalg.norm(np.subtract(point_rx, antenna_img))  # distance traveled
        self.LOS = 0  # = 1 if the ray is from LOS
        self.tau = self.d/self.c  # propagation delay
        self.ground_R = 0  # =1 if the ray has ground reflection
        self.theta_i_ground = None  # incidence angle with the ground reflection
        self.nbr_R = 0  # nbr of reflection
        self.theta_i = 0  # sum of incidence angle








