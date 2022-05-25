import os

import numpy as np
from matplotlib import pyplot as plt, pyplot, transforms
from matplotlib.collections import PathCollection
from matplotlib.transforms import Affine2D

from Wall import Wall
from Antenna import Antenna
from Ray_Tracing import RayTracing
from utils import *
from tqdm import tqdm
import time
import warnings



class Main:
    def __init__(self, wall_list=None, tx=None, rx=None, street='s',
                 draw=1, simu='heatmap', resolution=1, impulse_response= 1):
        plt.figure(1)
        self.start = time.time()
        self.impulse_response = impulse_response
        self.resolution = resolution
        self.draw = draw
        self.simu = simu
        self.street = street
        if wall_list is not None:
            self.wall_list = wall_list
        else:
            self.initialize_wall()
        if tx is not None:
            self.tx = tx
        else:
            self.tx = Antenna(5, 95, 'tx')
        if rx is not None:
            self.rx = rx
        else:
            if simu == 'LOS':
                self.rx = Antenna(50, 5, 'rx')
            elif simu == 'diff':
                self.rx = Antenna(55, 5, 'rx')
                #self.rx = Antenna(40, 55, 'rx')
            elif self.simu == 'path_loss':
                NotImplementedError

            elif simu == 'heatmap':
                temp = []
                x_ = 15
                y_ = 15
                self.bound = [0.5-x_, 50.5+x_, 100.5+y_, 0.5-y_]
                self.d = [50 + 2*x_, 100 + 2*y_]
                nbr_rx = (round(self.d[0]/self.resolution) * round(self.d[1]/self.resolution))
                start_1 = time.time()
                with tqdm(total=nbr_rx)as pbar:
                    count = 0
                    for i in np.arange(self.bound[0], self.bound[1], self.resolution):
                        for j in np.arange(self.bound[2], self.bound[3], -self.resolution):
                            temp.append(Antenna(i, j, 'rx', plot=False))
                            pbar.update()
                            pbar.set_description("Generate rx %i" % count)
                            count += 1
                    print("\n generating rx took is {} min {} seconds\n".format(round((time.time() - start_1)/60), round((time.time() - start_1)%(60))))
                    self.rx = temp
                    self.draw = 0
        self.rays = None
        self.compute_ray()
        self.show()

    def initialize_wall(self):
        data = np.reshape(np.loadtxt('grand_place.txt', delimiter=None, dtype=int), (-1, 5))
        temp = []
        for i in range(data.shape[0]):
            if data[i, 4] == 1:
                street = 'c'
            else:
                street = 's'
            temp.append(Wall(data[i, 0], -data[i, 1]+100, data[i, 2], -data[i, 3]+100, street))
        self.wall_list = temp

    def check_in_map(self, rx):
        in_map = False
        rx = [rx.pos_x, rx.pos_y]
        if 0 < rx[0] < 50:
            if 40 < rx[0] < 50:
                if -28 < rx[1] < 128:
                    in_map = True
            elif 0 < rx[1] < 100:
                in_map = True

        elif -28 < rx[0] < 0:
            if 0 < rx[1] < 10 or 65 < rx[1] < 75 or 90 < rx[1] < 100:
                in_map = True
        elif 50 < rx[0] < 78:
            if 25 < rx[1] < 35 or 65 < rx[1] < 75 or 90 < rx[1] < 100:
                in_map = True
        else:
            in_map =False
        if np.linalg.norm(np.subtract(self.tx.P,rx)) < 10 :
            in_map = False
        return in_map

    def compute_ray(self):
        if self.simu != 'heatmap':
            obj = RayTracing(self.wall_list, self.tx, self.rx, street=self.street, draw=self.draw)
            obj.ray_tracer()
            self.rays = obj.get_rays()
        else:
            nbr_rx = len(self.rx)
            self.big_mat = np.zeros((nbr_rx, 6))#P_RX_dBm, SNR_dB, K_dB, delay_spread, x, y
            start = time.time()
            with tqdm(total=nbr_rx) as pbar:
                for i in range(nbr_rx):
                    if self.check_in_map(self.rx[i]):
                        try:
                            obj = RayTracing(self.wall_list, self.tx, self.rx[i], street=self.street, draw=self.draw)
                            obj.ray_tracer()
                            self.rays = obj.get_rays()
                            [P_RX_dBm, SNR_dB, K_dB, delay_spread] = calculation_received_power(self.rays)
                            self.big_mat[i, :] = [P_RX_dBm, SNR_dB, K_dB, delay_spread, self.rx[i].pos_x, self.rx[i].pos_y]
                        except:
                            self.big_mat[i, :] = [None, None, None, None, self.rx[i].pos_x, self.rx[i].pos_y]
                    else:
                        self.big_mat[i, :] = [None, None, None, None, self.rx[i].pos_x, self.rx[i].pos_y]

                    pbar.set_description("Compute rays for wall %i" % i)
                    pbar.update()
            np.save('big_mat', self.big_mat)
            self.big_mat = np.load('big_mat.npy')
            print("\ncomputing rays take is {} min {} seconds\n".format(round((time.time() - start)/60), round((time.time() - start)%(60))))

    def show(self):
        if self.simu !='heatmap':
            plt.title('Ray Tracing for tx ({},{}) and rx ({},{})'.format(self.tx.pos_x,
                                                                         self.tx.pos_y,
                                                                         self.rx.pos_x,
                                                                         self.rx.pos_y))
            [P_RX_dBm, SNR_dB, K_dB, delay_spread] = calculation_received_power(self.rays, impulse_response = self.impulse_response)
            plt.text(51,80,'P_RX_dBm = ' + str(round(P_RX_dBm, 2)) + '\n' +
                      'SNR_dB =' + str(round(SNR_dB, 2)) + '\n' +
                      'K_dB = ' + ['-' if K_dB is None else str(round(K_dB, 2))][0] + '\n' +
                      'delay_spread = ' + str(round(delay_spread, 10)) + '\n', color='red')
            print(" \ntotal time taken is is {} min {} seconds\n".format(round((time.time() - self.start)/60), round((time.time() - self.start)%(60))))

            plt.show()
        else:
            start3 = time.time()
            with tqdm(total=4) as pbar:
                idx = 0
                for feat in ['P_RX_dBm', 'SNR_dB', 'K_dB', 'delay_spread']:
                    pbar.update()
                    pbar.set_description("Generate heatmap for %s" % feat)
                    try:
                        data = np.reshape(self.big_mat[:, idx], (round(self.d[0]/self.resolution),-1 ))#round(self.d[1]/self.resolution)
                    except ValueError:
                        data = np.reshape(self.big_mat[:, idx], (-1, -round(self.d[1]/self.resolution)))
                    fig, ax = plt.subplots()
                    im = ax.imshow(data.T)
                    plt.colorbar(im, ax=ax)
                    plt.title('heatmap for ' + feat)
                    try:
                        path = 'output/' + 'res =' + str(self.resolution) + '/'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        plt.savefig(path+ feat)
                    except:
                        path = 'output/' + 'resx100 =' + str(round(self.resolution*100)) + '/'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        plt.savefig(path + feat)
                    plt.show(block=False)
                    idx += 1
            print("\nplotting took is {} min {} seconds\n".format(round((time.time() - start3)/60), round((time.time() - start3)%(60*60))))
            print("\ntotal time taken is {} min {} seconds\n".format(round((time.time() - self.start) / 60),
                                                                     round((time.time() - self.start) % 60)))
            # plt.show()





#Main(simu='diff')
Main(simu='LOS')
#Main(simu='heatmap', resolution=1)

