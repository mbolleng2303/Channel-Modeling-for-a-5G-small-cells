import numpy as np
import math
from Ray import Ray
from matplotlib import pyplot as plt
'-----defining constant-------'
pi = math.pi
freq = 26*10**9
c = 3*10**8
omega = 2*pi*freq
beta = omega/c
Lambda = 2*pi/beta
epsilon_r = 5
R_ar = 71
R_al = 0
R_a = R_ar + R_al
eta = 1
directivity = 1.7
EIRP = 0.25
T0 = 293.15
k = 1.379*10**(-23)
B = 200*10**6
L_TX = 1
F_dB = 12
impulse_response = 1
SNR_dB_target = 5
G_TX = eta * directivity
P_TX = EIRP * L_TX / G_TX


def compute_rice_factor(h_e, E, LOS):
    maximum = LOS.max()
    index_LOS = np.where(LOS == maximum)[0][0]
    if maximum:
        V_oc = h_e*E # TODO
        P_RX = np.abs(V_oc)** 2 / (8 * R_a)
        a_n = np.sqrt(P_RX / P_TX)
        a_0 = a_n[index_LOS]
        a_i = np.array(a_n)
        a_i = np.delete(a_i, index_LOS)
        K = a_0**2 / np.sum(a_i**2)
        K_dB = 10 * np.log10(K)
    else :
        K_dB = None
        a_n = None
    return K_dB, a_n


def compute_SNR(P_RX):
    P_RX_dBW = 10 * np.log10(P_RX / 1)
    SNR_dB = P_RX_dBW - F_dB - 10 * np.log10(k * T0 * B)
    return SNR_dB


def compute_electric_field(ray):
    d = ray.d
    gamma = ray.gamma
    D = ray.D
    G_TX = eta * directivity
    if ray.ground_R :
        theta_i_transmitter = 180 - ray.theta_i_ground
        G_TX = 1.7 * np.sin(np.deg2rad(theta_i_transmitter)) ** 3
    E = gamma * D * np.sqrt(60 * G_TX * P_TX) * np.exp(-1j * beta * d) / d
    G_TX = eta * directivity
    return E


def compute_Lm():
    P_tx_dBm = 10 * np.log10(P_TX/0.001)
    Lm = P_tx_dBm - 10 * np.log10(k * T0 * B/0.001) - F_dB - SNR_dB_target
    return Lm


def calculation_h_e(ray):
    if ray.ground_R:
        theta_i_receiver = 180 - ray.theta_i_ground
        h_e = -Lambda/pi * np.cos(np.deg2rad(90 * np.cos(np.deg2rad(theta_i_receiver)))) / np.sin(np.deg2rad(theta_i_receiver))**2
    else:
        h_e = -Lambda / pi
    return h_e


def channel_impulse_response(rays, a_n, dB = True):
    """Physical impulse response"""
    plt.figure()
    tau_list = np.zeros(len(rays))
    phi_list = np.zeros(len(rays))
    h = np.zeros(len(rays)).astype(np.complex)
    for k in range(len(rays)):
        tau_list[k] = rays[k].tau
        phi_list[k] = rays[k].theta_i*pi/180
        h[k] = (a_n[k] * np.exp(1j * phi_list[k]) * np.exp(-2 * 1j * pi * freq * tau_list[k]))
    if dB:
        h_dB = 10*np.log10(np.abs(h))
    else:
        h_dB = np.abs(h)
    tau_list_ns = tau_list*10**9
    plt.stem(tau_list_ns, h_dB)
    plt.xlabel('tau [ns]')
    plt.ylabel('h(tau) [dB]')
    plt.gca().invert_yaxis()
    plt.title('Physical impulse Response')
    plt.savefig('output/Physical_impulse_Response.png')
    plt.show(block=False)

    """TDL impulse response"""
    plt.figure()
    delay_spread = tau_list.max() - tau_list.min()
    delta_fc = 1 / delay_spread
    BW = 200*10**6  # delta_fc/3
    delta_tau = 1 / 2*BW
    L = np.ceil(delay_spread/delta_tau)
    h_l_list = np.zeros((int(L), h.shape[0])).astype(np.complex)
    for l in range(int(L)):
        h_l_list[l, :] = h*np.sinc(2*BW*(tau_list-l*delta_tau))  # TODO: check with sacha
    h_l = np.sum(h_l_list, 1)
    if dB:
        h_l_dB = 10*np.log10(np.abs(h_l))
    else :
        h_l_dB = np.abs(h_l)
    t = 10**9 * delta_tau * np.arange(L)
    plt.stem(t, h_l_dB)
    plt.xlabel('tau [ns]')
    plt.ylabel('h(tau, t) [dB]')
    plt.gca().invert_yaxis()
    plt.title('TDL impulse Response for bandwidth = {} MHz'.format(BW/10**6))
    plt.savefig('output/TDL_impulse_Response.png')
    plt.show(block=False)


def compute_properties(rays, impulse_response=0):
    h_e = np.zeros(len(rays))
    E = np.zeros(len(rays)).astype(np.complex)
    LOS = np.zeros(len(rays))
    delay = np.zeros(len(rays))
    for k in range(len(rays)):
        E[k] = compute_electric_field(rays[k]) # TODO
        h_e[k] = calculation_h_e(rays[k])
        delay[k] = rays[k].tau
        if rays[k].LOS:
            LOS[k] = 1
    tau_min = delay.min()
    tau_max = delay.max()
    delay_spread = abs(tau_min-tau_max)
    K_dB, a_n = compute_rice_factor(h_e, E, LOS)
    if impulse_response:
        channel_impulse_response(rays, a_n, E)
    V_oc = np.dot(h_e, E)
    P_RX = np.abs(V_oc) ** 2 / (8 * R_a)
    P_RX_dBm = 10 * np.log10(P_RX / 0.001)
    SNR_dB = compute_SNR(P_RX)
    return [P_RX_dBm, SNR_dB, K_dB, delay_spread]


def compute_theta_i(wall, point1, point2):
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
        impact, pt_impact = Impact(wall, tx.P, rx.P)
        if impact:
            LOS = 0
    if LOS:
        ray_LOS = Ray(tx.P, rx.P)
        if draw:
            plt.plot([tx.pos_x, rx.pos_x], [tx.pos_y, rx.pos_y], color='red')
    return ray_LOS, LOS


def check_intersection(wall_list, point1, point2):
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
    h = 2
    LOS_d = np.linalg.norm(np.subtract(rx_pos, tx_pos))
    distance = 2 * np.sqrt(h ** 2 + (LOS_d / 2) ** 2)
    ray_ground.d = distance
    ray_ground.tau = distance / c
    theta_i = np.rad2deg(np.arctan(LOS_d / (2 * h)))
    gamma_para = compute_gamma_para(theta_i)
    ray_ground.gamma = ray_ground.gamma * gamma_para
    ray_ground.ground_R = 1
    ray_ground.theta_i_ground = theta_i
    ray_ground.theta_i = theta_i
    pt_reflection = [(rx_pos[0] + tx_pos[0]) / 2, (rx_pos[1] + tx_pos[1]) / 2]
    if draw:
        plt.scatter(pt_reflection[0], pt_reflection[1], marker='X')
        plt.plot([rx_pos[0], tx_pos[0]], [rx_pos[1], tx_pos[1]], color='red')
    return ray_ground


def compute_gamma_para(theta_i):
    num = np.cos(np.deg2rad(theta_i)) - (1 / np.sqrt(epsilon_r)) * np.sqrt(1 - (np.sin(np.deg2rad(theta_i)) ** 2 / epsilon_r))
    den = np.cos(np.deg2rad(theta_i)) + (1 / np.sqrt(epsilon_r)) * np.sqrt(1 - (np.sin(np.deg2rad(theta_i)) ** 2 / epsilon_r))
    gamma_para = num / den
    return gamma_para


def compute_gamma_perp(theta_i):
    num = np.cos(np.deg2rad(theta_i)) - np.sqrt(epsilon_r) * np.sqrt(1 - (np.sin(np.deg2rad(theta_i)) ** 2 / epsilon_r))
    den = np.cos(np.deg2rad(theta_i)) + np.sqrt(epsilon_r) * np.sqrt(1 - (np.sin(np.deg2rad(theta_i)) ** 2 / epsilon_r))
    gamma_perp = num / den
    return gamma_perp


def compute_coef_diffraction(tx_pos, corner, rx_pos):
    v = np.subtract(rx_pos, tx_pos)
    r_los = np.linalg.norm(v)
    r_ke = np.linalg.norm(np.subtract(corner, tx_pos)) + np.linalg.norm(np.subtract(rx_pos, corner))
    delta_r = r_ke - r_los
    nu = np.sqrt((2 / pi) * beta * delta_r)
    arg_F = -(pi / 4) - (pi * (nu ** 2)) / 2
    F_square_dB = -6.9 - 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1)
    F_square = 10 ** (F_square_dB / 20)
    mod_F = np.sqrt(F_square)
    D = mod_F * np.exp(1j * arg_F)
    return D, r_ke




