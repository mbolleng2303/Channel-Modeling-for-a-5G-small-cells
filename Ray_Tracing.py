import numpy as np
from matplotlib import pyplot as plt
from Wall import Wall
from Antenna import Antenna
from utils import *
pi = math.pi
freq = 26*10**9  # carrier frequency (26 GHz)
c = 3*10**8  # speed of light (m/s)
omega = 2*pi*freq  # carrier angular frequency (rad/s)
beta = omega/c  # wave number (rad/m)
Lambda = 2*pi/beta  # wave length(m)
epsilon_r = 5  # relative permittivity


class RayTracing:

    def __init__(self, wall_list, tx, rx, street='s', draw=1):
        self.wall_list = wall_list
        self.tx = tx
        self.rx = rx
        self.street = street
        self.draw = draw
        self.rays = []

    def ray_tracer(self):
        wall_list = self.wall_list
        tx = self.tx
        rx = self.rx
        draw = self.draw
        street = self.street
        'LOS'
        ray_LOS, LOS = line_of_sight(self.wall_list, tx, rx, draw)
        if LOS:
            ray_LOS.LOS = 1
            self.rays.append(ray_LOS)
        'Ground reflection'
        if LOS:
            ray_ground = ground_reflexion(rx.P, tx.P, draw)
            self.rays.append(ray_ground)
        'wall reflection in grand place'
        if street == 's':
            'One reflexion'
            for k in range(np.array(wall_list).shape[0]):
                wall = wall_list[k]
                antenna_image1 = antenna_image(wall, tx.P)
                impact1, pt_impact1 = Impact(wall, antenna_image1, rx.P)
                if impact1:
                    intersection_1a = check_intersection(wall_list, tx.P, pt_impact1)
                    intersection_1b = check_intersection(wall_list, pt_impact1, rx.P)
                    intersection_1 = intersection_1a + intersection_1b
                    if not intersection_1:
                        ray = Ray(antenna_image1, rx.P)
                        theta_i = compute_theta_i(wall, antenna_image1, rx.P)
                        gamma_perp = compute_gamma_perp(theta_i)
                        ray.gamma = ray.gamma * gamma_perp
                        ray.theta_i = ray.theta_i + theta_i
                        ray.nbr_R = 1
                        if draw:
                            plt.plot([tx.pos_x, pt_impact1[0], rx.pos_x], [tx.pos_y, pt_impact1[1], rx.pos_y], color='blue')
                        self.rays.append(ray)
                'Second reflexion'
                for i in range(np.array(wall_list).shape[0]):
                    wall2 = wall_list[i]
                    antenna_image2 = antenna_image(wall2, antenna_image1)
                    impact2, pt_impact2 = Impact(wall2, antenna_image2, rx.P)
                    if impact2:
                        impact3, pt_impact3 = Impact(wall, antenna_image1, pt_impact2)
                        if impact3:
                            intersection_2a = check_intersection(wall_list, tx.P, pt_impact3)
                            intersection_2b = check_intersection(wall_list, pt_impact3, pt_impact2)
                            intersection_2c = check_intersection(wall_list, pt_impact2, rx.P)
                            intersection_2 = intersection_2a + intersection_2b + intersection_2c
                            if not intersection_2:
                                ray = Ray(antenna_image2, rx.P)
                                ray.nbr_R = 2
                                'first reflexion'
                                theta_i = compute_theta_i(wall, antenna_image1, pt_impact2)
                                gamma_perp = compute_gamma_perp(theta_i)
                                ray.gamma = ray.gamma * gamma_perp
                                ray.theta_i = ray.theta_i + theta_i
                                'second reflexion'
                                theta_i = compute_theta_i(wall2, antenna_image2, rx.P)
                                gamma_perp = compute_gamma_perp(theta_i)
                                ray.gamma = ray.gamma * gamma_perp
                                ray.theta_i = ray.theta_i + theta_i
                                if draw:
                                    plt.plot([tx.pos_x, pt_impact3[0], pt_impact2[0], rx.pos_x], [tx.pos_y, pt_impact3[1], pt_impact2[1], rx.pos_y], color='green')
                                self.rays.append(ray)
        '''if not LOS:
            'Diffraction'
            street_walls = []
            for k in range(np.array(wall_list).shape[0]):
                wall = wall_list[k]
                if wall.street != street:
                    street_walls.append(wall)
            r = np.zeros_like(np.array(street_walls))
            for j in range(r.shape[0]):
                v = np.linalg.norm(np.subtract(street_walls[j].P2, rx.P))
                r[j] = v
            I = np.where(r == np.amin(r))[0][0]
            ray = Ray(tx.P, rx.P)
            street_corner = street_walls[I].P2
            coefD, r_ke = compute_coef_diffraction(tx.P, street_corner, rx.P)
            ray.d = r_ke
            ray.D = ray.D * coefD
            if draw:
                plt.plot([tx.pos_x, street_walls[I].x2, rx.pos_x], [tx.pos_y, street_walls[I].y2, rx.pos_y], color = 'yellow')
            self.rays.append(ray)'''

    def get_rays(self):
        return self.rays







