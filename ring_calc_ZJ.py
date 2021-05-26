import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import comb
import scipy.integrate as integrate
import numpy as np
from classes.plotting import *
from parameters_functions import *

N_block = 1000
s = 10
theta_0 = 0# / 180 * np.pi
amp_c_c = 1.5
j_max = 1.2 * amp_c_c
j_min = -0.7 * amp_c_c
c = 0

def t_fun_sig(current):
    """transfer function rate model"""
    beta_s = 12.0
    h_s = 0.1
    sig = 0.5 * (1 + np.tanh(beta_s * (current - h_s)))
    return sig


def chebyshev_coefficient(l, n):
    """compute the chebyshev coefficient A_l;.: \cos(\theta)^n = A_l * \cos(l*\theta)"""
    A_l = 0
    if l == 0 and (n % 2) == 0:
        A_l = 1 / 2 * 2 ** (1 - n) * comb(n, n / 2)
    elif (n - l) % 2 == 0:
        A_l = 2 ** (1 - n) * comb(n, (n - l) / 2)
    return A_l

def func(q):
    """function whose zeros are to be solved; the zeros are the order parameters"""
    q_next = np.zeros(2 * (s + 1) + 1)

    def input(theta):  # input current to the neuron with PF = theta
        ret = j_min * q[-1] + c * (np.cos(theta - theta_0)) ** s
        for l in range(s + 1):
            ret = ret + (j_max-j_min) * np.cos(l*(theta-theta_0)) * q[l] + (j_max-j_min) * np.sin(
                l*(theta-theta_0)) * q[l+s+1]
        return ret

    for l in range(s + 1): # paremeters q, from index 0 to s+1
        f = lambda theta: np.cos(l * (theta - theta_0)) * t_fun_sig(input(theta))  # integrand
        integral = integrate.quad(f, -np.pi/2, np.pi/2)
        result = integral[0]
        q_next[l] = q[l] - A[l] / np.pi * result

    for l in range(s + 1): # parameters p, from index s+1 to 2s+2
        f = lambda theta: np.sin(l * (theta - theta_0)) * t_fun_sig(input(theta))  # integrand
        integral = integrate.quad(f, -np.pi/2, np.pi/2)
        result = integral[0]
        q_next[l+s+1] = q[l+s+1] - A[l] / np.pi * result

    f = lambda theta: t_fun_sig(input(theta))  # integrand
    integral = integrate.quad(f, -np.pi / 2, np.pi / 2)
    result = integral[0]
    q_next[-1] = q[-1] - result / np.pi # parameters \bar{r}

    return q_next

# q = np.zeros(2*(s+1)+1) # (s+1) q's and (s+1) p's and \bar(r) Here, we use q for all paras
q = np.array([ 1.09372599e-02, -7.09510433e-03,  7.15245074e-03, -4.96866239e-03,
       -4.05134635e-04, -2.50888318e-03, -1.83404505e-03, -1.55287468e-03,
       -7.10068383e-04, -1.59027612e-04, -8.91766974e-05, -1.01146644e-09,
        1.76526726e-04,  1.08565288e-03,  4.48599590e-06,  3.05822431e-04,
       -9.44890172e-05,  1.40216347e-04, -1.97780974e-04, -9.19992578e-05,
       -3.70265067e-05,  1.08685192e-05,  6.53532358e-02])

A = np.zeros(s + 1) #chebyshev coefficients
for i in range(s + 1):
    A[i] = chebyshev_coefficient(i, s)
    # if A[i] != 0:
    #     q[i] = 0.01
    #     q[i+s+1] = 0.01

root = fsolve(func, q)
# print(f'The solution is {root}')
# print(f'Is the root closed to a solution? {np.isclose(func(root), np.zeros(2 * s + 2 + 1))}')

# recover the rates variable
theta_range = np.linspace(-np.pi / 2., np.pi / 2., N_block)
rates_MFT = np.zeros(N_block)
for i, theta in zip(range(N_block), theta_range):
    input_ss = j_min * root[-1] + c * (np.cos(theta - theta_0)) ** s
    for l in range(s + 1):
        input_ss = input_ss + (j_max - j_min) * np.cos(l * (theta - theta_0)) * root[l] + (j_max - j_min) * np.sin(
            l * (theta - theta_0)) * root[l + s + 1]
    rates_MFT[i] = t_fun_sig(input_ss)
# fake_dyn = np.tile(rates_MFT, [20000, 1]) #in order to use Ulises' plotting code
# modelparams = model_parameters()
# plot_dynamics_cortex_trial(fake_dyn, modelparams)
plt.plot(theta_range, rates_MFT)
plt.show()