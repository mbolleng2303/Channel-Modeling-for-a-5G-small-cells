import os
import scipy
from Wall import Wall
from Antenna import Antenna
from Ray_Tracing import RayTracing
from utils import *
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')


class Main:
    def __init__(self, wall_list=None, tx=None, rx=None, street='s',
                 draw=1, simu='heatmap', resolution=1, impulse_response=0, show=False, TX = [[0,0],[0,0]]):
        if simu != 'TX':
            plt.figure()
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
                self.rx = Antenna(25, 50, 'rx')
            elif simu == 'diff':
                self.rx = Antenna(75, 5, 'rx')
                #self.rx = Antenna(40, 55, 'rx')
            elif self.simu == 'plot_1D' or self.simu == 'path_loss':
                temp = []
                d = []
                x_ = np.arange(5, 50, self.resolution)
                for x in x_:
                    y = 105 - (90/45)*x
                    d.append(np.linalg.norm([x-self.tx.pos_x, y-self.tx.pos_y]))
                    temp.append(Antenna(x, y, 'rx', plot=True))
                self.rx = temp
                self.d = d
                self.draw = 0
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
            elif simu == 'TX':
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
                    self.tx = [Antenna(TX[0][0], TX[0][1], 'tx'), Antenna(TX[1][0], TX[1][1], 'tx')]
                    self.draw = 0
        self.rays = None
        self.compute_ray()
        self.show()
        if show:
            plt.show()

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
                if 0 < rx[1] < 128:
                    in_map = True
            elif 0 < rx[1] < 100:
                in_map = True

        elif -28 < rx[0] < 0:
            if 0 < rx[1] < 10 or 25 < rx[1] < 35 or 90 < rx[1] < 100:
                in_map = True
        elif 50 < rx[0] < 78:
            if 0 < rx[1] < 10 or 25 < rx[1] < 35 or 65 < rx[1] < 75:
                in_map = True
        else:
            in_map =False
        if self.simu =='TX':
            for i in range(len(self.tx)):
                if np.linalg.norm(np.subtract(self.tx[i].P, rx)) < 10:
                    in_map = False
        else :
            if np.linalg.norm(np.subtract(self.tx.P, rx)) < 10:
                in_map = False

        return in_map

    def compute_ray(self):
        if self.simu == 'LOS' or self.simu == 'diff':
            obj = RayTracing(self.wall_list, self.tx, self.rx, street=self.street, draw=self.draw)
            obj.ray_tracer()
            self.rays = obj.get_rays()
        elif self.simu == 'plot_1D' or self.simu == 'heatmap' or self.simu == 'path_loss':
            nbr_rx = len(self.rx)
            self.big_mat = np.zeros((nbr_rx, 6))  # P_RX_dBm, SNR_dB, K_dB, delay_spread, x, y
            start = time.time()
            with tqdm(total=nbr_rx) as pbar:
                for i in range(nbr_rx):
                    if self.check_in_map(self.rx[i]):
                        try:
                            obj = RayTracing(self.wall_list, self.tx, self.rx[i], street=self.street, draw=self.draw)
                            obj.ray_tracer()
                            self.rays = obj.get_rays()
                            [P_RX_dBm, SNR_dB, K_dB, delay_spread] = compute_properties(self.rays)
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
        else:
            nbr_rx = len(self.rx)
            nbr_tx = len(self.tx)
            self.big_mat = np.zeros((nbr_rx, 4))  # P_RX_dBm, SNR_dB, K_dB, delay_spread, x, y
            start = time.time()

            with tqdm(total=nbr_rx*nbr_tx) as pbar:
                for i in range(nbr_rx):
                    ray_ = []
                    for j in range(nbr_tx):
                        if self.check_in_map(self.rx[i]) and self.big_mat[i, 0] != None:

                            obj = RayTracing(self.wall_list, self.tx[j], self.rx[i], street=self.street, draw=self.draw)
                            obj.ray_tracer()
                            self.rays = obj.get_rays()
                            for k in range(len(self.rays)):
                                ray_.append(self.rays[k])
                            if j == nbr_tx-1 :
                                [P_RX_dBm, SNR_dB, K_dB, delay_spread] = compute_properties(np.reshape(np.array(ray_), -1))
                                self.big_mat[i, :] = [P_RX_dBm, SNR_dB, K_dB, delay_spread]


                        else:
                            self.big_mat[i, :] = [None, None, None, None]

                        pbar.set_description("Compute rays for wall %i" % i)
                        pbar.update()
            np.save('big_mat', self.big_mat)
            idx = np.where(self.big_mat[:, 1] == self.big_mat[:, 1])
            pos = np.where(self.big_mat[idx, 1] >= 5)
            pos_ratio = len(self.big_mat[pos,1].T)
            ratio = pos_ratio /len(self.big_mat[idx,1].T)
            self.SNR_mean = np.mean(self.big_mat[idx, 1])
            self.ratio = ratio
            print("\ncomputing rays take is {} min {} seconds\n".format(round((time.time() - start) / 60),
                                                                        round((time.time() - start) % (60))))

    def show(self):
        if self.simu == 'diff' or self.simu == 'LOS':
            plt.title('Ray Tracing for tx ({},{}) and rx ({},{})'.format(self.tx.pos_x,
                                                                         self.tx.pos_y,
                                                                         self.rx.pos_x,
                                                                         self.rx.pos_y))
            [P_RX_dBm, SNR_dB, K_dB, delay_spread] = compute_properties(self.rays, impulse_response=self.impulse_response)
            plt.text(51, 80, 'P_RX_dBm = ' + str(round(P_RX_dBm, 2)) + '\n'+
                      'SNR_dB =' + str(round(SNR_dB, 2)) + '\n' +
                      'K_dB = ' + ['-' if K_dB is None else str(round(K_dB, 2))][0] + '\n' +
                      'delay_spread = ' + str(round(delay_spread, 10)) + '\n', color='red')
            print(" \ntotal time taken is is {} min {} seconds\n".format(round((time.time() - self.start)/60), round((time.time() - self.start)%(60))))

            plt.show(block=False)
        elif self.simu == 'plot_1D':
            idx = 0
            for feat in ['P_RX_dBm', 'SNR_dB', 'K_dB', 'delay_spread']:
                plt.figure()
                data = self.big_mat[:, idx]
                plt.plot(self.d, data)
                plt.title(feat + ' from tx = ({},{}) until ({},{})'.format(round(self.tx.pos_x, 1), round(self.tx.pos_y, 1), round(self.rx[-1].pos_x, 1), round(self.rx[-1].pos_y, 1)))
                plt.xlabel('distance [m]')
                plt.ylabel(feat)
                plt.savefig('output/' + feat + '_plot_1D.png')
                idx += 1
            plt.show(block=False)
        elif self.simu == 'path_loss':
            plt.figure()
            idx = np.where(self.big_mat[:, 0] == self.big_mat[:, 0])  # remove nan value
            P_RX_dBm = self.big_mat[idx, 0][0, :]
            self.d = np.array(self.d)[idx]
            [m, p] = np.polyfit(np.log10(self.d), P_RX_dBm, 1)
            fitted_P_RX_dBm = m*np.log10(self.d)+p
            plt.plot(np.log10(self.d), fitted_P_RX_dBm)
            plt.plot(np.log10(self.d), P_RX_dBm)
            plt.title('power receive  from tx = ({},{}) until ({},{})'.format(round(self.tx.pos_x, 1), round(self.tx.pos_y, 1),
                                                                       round(self.rx[-1].pos_x, 1),
                                                                       round(self.rx[-1].pos_y, 1)))
            plt.legend(['Fitted data', 'Original data'])
            plt.xlabel('log(d)')
            plt.ylabel('P_RX_dBm')
            plt.savefig('output/' + 'regression.png')
            plt.show(block=False)
            plt.figure()
            P_tx_dBm = 10 * np.log10(P_TX / 0.001)
            path_loss = P_tx_dBm - P_RX_dBm
            path_loss_linear = P_tx_dBm - fitted_P_RX_dBm
            plt.plot(np.log10(self.d), path_loss_linear)
            plt.plot(np.log10(self.d), path_loss)
            plt.legend(['Fitted data', 'Original data'])
            plt.title('path loss from tx = ({},{}) until ({},{})'.format(round(self.tx.pos_x, 1),
                                                                              round(self.tx.pos_y, 1),
                                                                              round(self.rx[-1].pos_x, 1),
                                                                              round(self.rx[-1].pos_y, 1)))
            plt.xlabel('log(d)')
            plt.ylabel('path_loss')
            plt.savefig('output/' + 'path_loss.png')
            plt.show(block=False)

            #PATH LOSS WITH GLOBAL FADING

            sigma_L = np.std(path_loss - path_loss_linear)
            plt.figure()
            plt.plot(np.log10(self.d), path_loss - path_loss_linear)
            plt.plot([np.log10(self.d).min(), np.log10(self.d).max()], [sigma_L, sigma_L], color='red')
            plt.plot([np.log10(self.d).min(), np.log10(self.d).max()], [-sigma_L, -sigma_L], color='red')
            plt.legend(['path_loss', 'Variability '])
            plt.title('Fadind variability')
            plt.xlabel('log(d)')
            plt.ylabel('Zero-mean power in dBm')
            plt.savefig('output/' + 'path_loss_variability.png')
            plt.show(block=False)
            plt.figure()
            L_sigma = np.log10(np.random.lognormal(0, sigma_L, len(path_loss)))
            path_loss_model = path_loss_linear + L_sigma
            plt.plot(np.log10(self.d), path_loss_model)
            plt.title("Path loss model with shadowing")
            plt.xlabel("log(d)")
            plt.ylabel("L [dB]")

            [m1, p1] = np.polyfit(np.log10(self.d), path_loss_model, 1)
            path_loss_linear_final = m1 * np.log10(self.d)+p1
            plt.plot(np.log10(self.d), path_loss_linear_final)

            #  Cell range as a function of connection probability
            n = m1/10
            Pr = np.arange(0.1, 1, 0.01)
            gamma = np.zeros(len(Pr))
            L_r = np.zeros(len(Pr))
            Lm = compute_Lm()
            for i in range(len(Pr)):
                gamma[i] = scipy.special.erfcinv((Pr[i])*2)*sigma_L*np.sqrt(2)
                L_r[i] = Lm + gamma[i]
            d_0 = self.d[0]
            L0_d0 = path_loss_linear[0]
            R = d_0 * 10**((L_r - L0_d0) / (10 * n))
            plt.show(block=False)
            plt.savefig('output/' + 'Path_loss_model_with_shadowing.png')
            plt.figure()
            plt.plot(Pr, R)
            plt.xlabel('Connection probability')
            plt.ylabel('Cell range (m)')
            plt.title('Cell range as a function or the connection probability')
            plt.savefig('output/' + 'Cell range as a function or the connection probability')
            plt.show(block=False)
            plt.figure()
            plt.plot(Pr, L_r)
            plt.xlabel('Connection probability')
            plt.ylabel('Path loss at cell edge L(R)')
            plt.title('Path loss at cell edge L(R) as a function or the connection probability')
            plt.savefig('output/' + 'Path loss at cell edge L(R) as a function or the connection probability')
            plt.show(block=False)
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
                    path = 'output/' + 'res =' + str(self.resolution) + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    plt.savefig(path + self.simu+ '_'+ feat)

                    plt.show(block=False)
                    idx += 1
            print("\nplotting took is {} min {} seconds\n".format(round((time.time() - start3)/60), round((time.time() - start3)%(60*60))))
            print("\ntotal time taken is {} min {} seconds\n".format(round((time.time() - self.start) / 60),
                                                                     round((time.time() - self.start) % 60)))


resolution = 0.3
Main(simu='LOS', impulse_response=1)
Main(simu='plot_1D', resolution=resolution)
Main(simu='heatmap', resolution=resolution)
Main(simu='path_loss', resolution=resolution)
Main(simu='diff')
Main(simu='plot_1D', resolution=resolution)
Main(simu='TX', resolution=resolution, TX=[[25, 80], [25, 20]])
plt.show()


# Brute force approach
'''best_snr = [-1000]
for x1 in range(1, 49, resolution):
    for y1 in range(1, 99, resolution):
        for x2 in range(1, 49, resolution):
            for y2 in range(1, 99, resolution):
                if np.linalg.norm([abs(x2-x1), abs(y2-y1)]) >= 10:
                    simu = Main(simu='TX', resolution=resolution, TX=[[x1, y1], [x2, y2]])
                    snr = simu.ratio
                    if snr > best_snr[0]:
                        best_snr = [snr, x1, y1, x2, y2]
                        print(best_snr)
np.save('best_snr', best_snr)'''





