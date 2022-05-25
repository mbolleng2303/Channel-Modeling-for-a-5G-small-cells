import numpy as np
import math
from Ray import Ray
from matplotlib import pyplot as plt
'-----defining constant-------'
pi = math.pi
freq = 26*10**9  # carrier frequency (26 GHz)
c = 3*10**8  # speed of light (m/s)
omega = 2*pi*freq  # carrier angular frequency (rad/s)
beta = omega/c  # wave number (rad/m)
Lambda = 2*pi/beta  # wave length(m)
epsilon_r = 4  # relative permittivity
R_a = 71
eta = 1
directivity = 1.7
EIRP = 0.25
T0 = 293.15
k = 1.379*10**(-23)
B = 200*10**6
L_TX = 1
F_dB = 12
impulse_response = 1

def calculation_rice_factor(h_e,E,LOS_vector):
    maximum = LOS_vector.max()
    index_LOS = np.where(LOS_vector==maximum)[0][0]
    if maximum:
        P_TX,_ = calculation_transmitted_power()
        V_oc = h_e*E # TODO
        P_RX = np.abs(V_oc)** 2 / (8 * R_a)
        attenuation_factor = np.sqrt(P_RX/ P_TX)
        a_0 = attenuation_factor[index_LOS]
        a_i = np.array(attenuation_factor)
        a_i = np.delete(a_i, index_LOS)
        K = a_0**2 / np.sum(a_i**2)
        K_dB = 10 * np.log10(K)
    else :
        K_dB = None
        attenuation_factor = None
    return K_dB, attenuation_factor


def calculation_transmitted_power():

    G_TX = calculation_transmitter_antenna_gain()
    P_TX = EIRP * L_TX / G_TX
    return P_TX, G_TX


def calculation_SNR(P_RX):
    P_RX_dBW = 10 * np.log10(P_RX / 1)
    SNR_dB = P_RX_dBW - F_dB - 10 * np.log10(k * T0 * B)
    return SNR_dB


def calculation_transmitter_antenna_gain():
    G_TX = eta * directivity
    return G_TX


def calculation_electric_field(ray):
    P_TX, G_TX = calculation_transmitted_power()
    d = ray.d
    gamma = ray.gamma
    D = ray.D
    if ray.ground_R :
        theta_i_transmitter = 180 - ray.theta_i_ground
        G_TX = 1.7 * np.sin(np.deg2rad(theta_i_transmitter)) ** 3
    E = gamma * D * np.sqrt(60 * G_TX * P_TX) * np.exp(-1j * beta * d) / d
    return E


def calculation_effective_height(ray):
    if ray.ground_R:
        theta_i_receiver = 180 - ray.theta_i_ground
        h_e = -Lambda/pi * np.cos(np.deg2rad(90 * np.cos(np.deg2rad(theta_i_receiver)))) / np.sin(np.deg2rad(theta_i_receiver))**2
    else:
        h_e = -Lambda / pi
    return h_e


def sum_duplicate(tau_list, h):

    return h, tau_list


def channel_impulse_response(rays, attenuation_factor,E, dB = True):
    """Physical impulse response"""
    tau_list = np.zeros(len(rays))
    phi_list = np.zeros(len(rays))
    h = np.zeros(len(rays)).astype(np.complex)
    for k in range(len(rays)):
        tau_list[k] = rays[k].tau
        phi_list[k] = rays[k].theta_i*pi/180
        h[k] = (attenuation_factor[k]*np.exp(1j*phi_list[k])*np.exp(-2*1j*pi*freq*tau_list[k]))
    h_bis, tau_list_bis = sum_duplicate(tau_list, h) #TODO
    if dB:
        h_bis_dB = 10*np.log10(np.abs(h_bis))
    else:
        h_bis_dB = np.abs(h_bis)
    tau_list_ns = tau_list_bis*10**9
    plt.figure(2)
    #h_bis_dB=abs(E)
    plt.stem(tau_list_ns, h_bis_dB)# TODO : abs(E)
    plt.xlabel('tau [ns]')
    plt.ylabel('h(tau) [dB]')
    plt.title('Physical impulse Response')

    """TDL impulse response"""

    delay_spread = tau_list.max() - tau_list.min()
    delta_fc = 1 / delay_spread
    BW_TDL = 200*10**6 #delta_fc # narrow band 200*10**6 #
    delta_tau_TDL = 1 / (2*BW_TDL)
    L = np.ceil(tau_list.max()/delta_tau_TDL)
    #h = E #TODO
    h_l_matrix = np.zeros((int(L + 1), h.shape[0])).astype(np.complex)
    for l in range(int(L+1)):
        h_l_matrix[l, :] = E*np.sinc(2*BW_TDL*(tau_list-l*delta_tau_TDL))  #TODO: h, (l+1)
    h_l = np.sum(h_l_matrix, 1)
    if dB:
        h_l_dB = 10*np.log10(np.abs(h_l))
    else :
        h_l_dB = np.abs(h_l)

    t_TDL = delta_tau_TDL * 10**9 * np.arange(L+1)
    base_value = h_l_dB.min() - 10
    max_value = h_l_dB.max() + 20
    plt.figure(3)
    plt.stem(t_TDL, h_l_dB)
    plt.xlabel('tau [ns]')
    plt.ylabel('h(tau, t) [dB]')
    plt.title('TDL impulse Response')
    plt.ylim([base_value, max_value])


    """TDL US  impulse response"""
    """BW = 1/delay_spread
    delta_tau = 1 / (BW)
    taps = np.zeros(tau_list.shape[0])
    for i in range(tau_list.shape[0]):
        taps[i] = np.ceil(tau_list[i]/delta_tau)
    h_l_bis, taps_bis = sum_duplicate(taps, h)
    if dB:
        h_l_bis_dB = 10*np.log10(np.abs(h_l_bis))
    else :
        h_l_bis_dB = np.abs(h_l_bis)
    plt.figure(4)
    plt.stem(taps_bis*delta_tau*10**9, h_l_bis_dB)
    plt.xlabel('tau [ns]')
    plt.ylabel('h(tau) [dB]')
    plt.title('TDL US impulse Response')"""


def calculation_received_power(rays, impulse_response = 0):
    R_ar = 73
    R_al = 0
    R_a = R_ar + R_al
    h_e = np.zeros(len(rays))
    E = np.zeros(len(rays)).astype(np.complex)
    LOS_vector = np.zeros(len(rays))
    list_delay = np.zeros(len(rays))
    for k in range(len(rays)):
        E[k] = calculation_electric_field(rays[k]) # TODO
        h_e[k] = calculation_effective_height(rays[k])
        list_delay[k] = rays[k].tau
        if rays[k].LOS:
            LOS_vector[k] = 1
    tau_min = list_delay.min()
    tau_max = list_delay.max()

    delay_spread = abs(tau_min-tau_max)
    K_dB, attenuation_factor = calculation_rice_factor(h_e, E, LOS_vector)
    if impulse_response:
        channel_impulse_response(rays, attenuation_factor, E)
    V_oc = np.dot(h_e, E)
    P_RX = np.abs(V_oc) ** 2 / (8 * R_a)
    P_RX_dBm = 10 * np.log10(P_RX / 0.001)
    SNR_dB = calculation_SNR(P_RX)
    return [P_RX_dBm, SNR_dB, K_dB, delay_spread]


def calculation_theta_i(wall, point1, point2):
    v = np.subtract(point1, point2)
    ang1 = np.arccos(np.dot(wall.n1, v) / (np.linalg.norm(wall.n1) * np.linalg.norm(v)))
    ang2 = np.arccos(np.dot(wall.n2, v) / (np.linalg.norm(wall.n2) * np.linalg.norm(v)))
    ang1 = np.rad2deg(ang1)
    ang2 = np.rad2deg(ang2)
    if ang1 < ang2:
        theta_i = ang1
    else:
        theta_i = ang2
    return theta_i


def line_of_sight(wall_list, tx, rx, draw):
    ray_LOS = []
    LOS = 1
    for wall in wall_list:
        impact, pt_impact = Impact(wall, tx.P, rx.P);
        if impact:
            LOS = 0
    if LOS:
        ray_LOS = Ray(tx.P, rx.P)
        if draw:
            plt.plot([tx.pos_x, rx.pos_x], [tx.pos_y, rx.pos_y], color='red')
    return ray_LOS, LOS


def check_intersection_wall_list(wall_list, point1, point2):
    intersection = 0
    for wall in wall_list:
        impact, pt_impact = Impact(wall, point1, point2)
        if impact:
            intersection = 1
    return intersection


def Impact(wall, point1, point2):
    x1 = wall.x1
    y1 = wall.y1
    x2 = wall.x2
    y2 = wall.y2

    x3 = point1[0]
    y3 = point1[1]
    v1 = (point2[0] - x3)
    v2 = (point2[1] - y3)
    x4 = x3 + v1
    y4 = y3 + v2
    pt_impact = [None, None]
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        impact = 0
    else:
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        if 0 < t < 1 and 0 < u < 1:
            impact = 1
            impact_x = x1 + t * (x2 - x1)
            impact_y = y1 + t * (y2 - y1)
            pt_impact = [impact_x ,impact_y]
            # plot([impact_x rx_x], [impact_y rx_y]);
            # scatter(impact_x, impact_y);
        else:
            impact = 0
    return impact, pt_impact


def antenna_image(wall, point):
    x1 = wall.x1
    y1 = wall.y1
    x2 = wall.x2
    y2 = wall.y2
    antenna_image = [None, None]
    if x1 == x2: # if the wall is vertical
        antenna_image[0] = 2 * x1 - point[0]
        antenna_image[1] = point[1]

    elif y1 == y2: # if the wall is horizontal
        antenna_image[0] = point[0]
        antenna_image[1] = 2 * y1 - point[1]

    return antenna_image


def ground_reflexion(rx_pos, tx_pos, draw):
    ray_ground = Ray(tx_pos, rx_pos)
    h = 2  # height of the user equipements(m)
    LOS_d = np.linalg.norm(np.subtract(rx_pos, tx_pos))
    distance = 2 * np.sqrt(h ** 2 + (LOS_d / 2) ** 2) # TODO : verify
    ray_ground.d = distance
    ray_ground.tau = distance / c
    theta_i = np.rad2deg(np.arctan(LOS_d / (2 * h)))#degree
    gamma_para = calculation_gamma_para(theta_i)
    ray_ground.gamma = ray_ground.gamma * gamma_para
    ray_ground.ground_R = 1
    ray_ground.theta_i_ground = theta_i
    ray_ground.theta_i = theta_i
    pt_reflection = [(rx_pos[0] + tx_pos[0]) / 2, (rx_pos[1] + tx_pos[1]) / 2]
    if draw:
        plt.scatter(pt_reflection[0], pt_reflection[1], marker='X')
    return ray_ground


def calculation_gamma_para(theta_i):
    num = np.cos(np.deg2rad(theta_i)) - (1 / np.sqrt(epsilon_r)) * np.sqrt(1 - (np.sin(np.deg2rad(theta_i)) ** 2 / epsilon_r))
    den = np.cos(np.deg2rad(theta_i)) + (1 / np.sqrt(epsilon_r)) * np.sqrt(1 - (np.sin(np.deg2rad(theta_i)) ** 2 / epsilon_r))
    gamma_para = num / den
    return gamma_para


def calculation_gamma_perp(theta_i):
    num = np.cos(np.deg2rad(theta_i)) - np.sqrt(epsilon_r) * np.sqrt(1 - (np.sin(np.deg2rad(theta_i)) ** 2 / epsilon_r))
    den = np.cos(np.deg2rad(theta_i)) + np.sqrt(epsilon_r) * np.sqrt(1 - (np.sin(np.deg2rad(theta_i)) ** 2 / epsilon_r))
    gamma_perp = num / den
    return gamma_perp


def calculation_coef_diffraction(tx_pos,corner,rx_pos):
    Delta_r, r_ke = calculation_Delta_r(tx_pos, corner, rx_pos)
    nu = calculation_nu(Delta_r)
    mod_F = calculation_module_F(nu)
    arg_F = calculation_argument_F(nu)
    coefD = mod_F * np.exp(1j * arg_F)
    return coefD, r_ke


def calculation_Delta_r(tx_pos,corner,rx_pos):
    v = np.subtract(rx_pos ,tx_pos)
    r_los = np.linalg.norm(v)
    r_ke = np.linalg.norm(np.subtract(corner, tx_pos)) + np.linalg.norm(np.subtract(rx_pos, corner))
    Delta_r = r_ke - r_los
    return Delta_r, r_ke


def calculation_nu(Delta_r):
    nu = np.sqrt((2 / pi) * beta * Delta_r)
    return nu


def calculation_module_F(nu):
    F_square_dB = -6.9 - 20*np.log10(np.sqrt((nu-0.1)**2 + 1) + nu - 0.1)
    F_square = 10 ** (F_square_dB / 10)
    mod_F = np.sqrt(F_square)
    return mod_F


def calculation_argument_F(nu):
    arg_f = -(pi/4)-(pi*(nu**2))/2
    return arg_f
