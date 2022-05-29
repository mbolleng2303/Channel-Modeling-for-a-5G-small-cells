import math
import numpy as np
pi = math.pi
freq = 26*10**9  # carrier frequency (26 GHz)
c = 3*10**8  # speed of light (m/s)
omega = 2*pi*freq  # carrier angular frequency (rad/s)
beta = omega/c  # wave number (rad/m)
Lambda = 2*pi/beta  # wave length(m)
epsilon_r = 5 # relative permittivity
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
def gamma(theta_i):
    num = np.cos(np.deg2rad(theta_i)) - (np.sqrt(epsilon_r)) * np.sqrt(1 - (np.sin(np.deg2rad(theta_i)) ** 2 / epsilon_r))
    den = np.cos(np.deg2rad(theta_i)) + (np.sqrt(epsilon_r)) * np.sqrt(1 - (np.sin(np.deg2rad(theta_i)) ** 2 / epsilon_r))
    gamma_para = num / den
    return gamma_para
#
he = -Lambda/pi
d = 121.6720
theta_i = np.rad2deg(np.arctan(45/100))
T = gamma(theta_i)
print(T)
D = -0.0661-0.0054*1j
E = D*np.sqrt(60*0.25) * np.exp(-1j*beta*d)/d#-0.0017 -0.0013*1j#-0.0196 + 0.0331*1j#2 * (0.0007 - 0.0129*1j) + - 0.0090 + 0.0182*1j
print(E)
V = E*he
print(V)
P_rx = (np.abs(V)**2)/(8*R_a)
print(P_rx)
P_rx_dBm = 10*np.log10(P_rx/0.001)
print(P_rx_dBm)

