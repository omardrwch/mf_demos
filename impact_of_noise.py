"""
This script tests the impact of additive noise on the estimation
of multifractal properties.

Precisely, let X_t be a multifractal random process; and let N_t be the 
noise. We analyze here the signal Y_t = X_t + sigma*N_t
aiming to see for which values of sigma the log-cumulants of Y_t are 
sufficiently close to the log-cumulants of X_t.
"""

import mfanalysis as mf
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from colored_noise import powerlaw_psd_gaussian
from scipy.signal import welch
from scipy.stats import linregress

plt.rcParams.update({'mathtext.default':  'regular', 'font.size': 16})

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def get_data_from_mat_file(filename):
    contents = loadmat(filename)
    return contents['data'][0]

def plot_psd(signal, fs = 100, name = '', f1 = 1.0, f2 = 10.0, nperseg = 1024):
    f, px = welch(signal, fs, scaling = 'spectrum', nperseg=nperseg)
    plt.figure()
    plt.loglog(f, px)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Power spectrum - ' + name)
    if (f1 is not None) and (f2 is not None):
        ff = f[ np.logical_and(f>=f1, f<=f2) ].copy()
        PP = px[ np.logical_and(f>=f1, f<=f2) ].copy()
        log_ff = np.log10(ff)
        log_PP = np.log10(PP)
        slope, intercept, r_value, p_value, std_err = linregress(log_ff,log_PP)
        log_PP_fit = slope*log_ff + intercept
        PP_fit    =  10.0**(log_PP_fit)
        plt.loglog(ff, PP_fit, label = 'beta=%f'%(slope))
        plt.legend()
    plt.grid()

#-------------------------------------------------------------------------------
# Load and normalize data, generate noise
#-------------------------------------------------------------------------------
# multifractal random walk (c_1=0.75, c_2=-0.05, N=32768)
# data_file = 'example_data/mrw07005n32768.mat'
data_file = 'example_data/S010.mat'


current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, data_file)
data = get_data_from_mat_file(data_file)

# normalize signal
data = data / data.std()

# generate noise
NOISE_TYPE = 'colored'
noise_params = {'beta':1.5, 'len':len(data), 'n_sigma':40}

if NOISE_TYPE == 'white':
    noise = np.random.normal(loc = 0, scale = 1.0, size=noise_params['len']) #  np.sin(np.arange(len(data))) 
elif NOISE_TYPE == 'colored':
    noise = powerlaw_psd_gaussian(noise_params['beta'], noise_params['len'])
    noise = noise/noise.std()

elif NOISE_TYPE == 'smooth':
    nn = np.arange(noise_params['len'])/noise_params['len'] - 0.5
    noise = nn**3.0
    noise = noise/noise.std()

elif NOISE_TYPE == 'dirac':
    noise = np.random.binomial(1, 5/noise_params['len'], noise_params['len'])
    noise *= 5
    #noise = noise/noise.std()


# vector of sigma^2 (variances)
sigma2 = np.linspace(0., 0.1, noise_params['n_sigma']) # np.array([0., 0.000001]) 


#-------------------------------------------------------------------------------
# MFA parameters
#-------------------------------------------------------------------------------
p_list = [2.0,  np.inf]

# Multifractal analysis object
mfa = mf.MFA()
mfa.wt_name = 'db3'
# mfa.p = np.inf
mfa.j1 = 4
mfa.j2 = 9
mfa.n_cumul = 3
mfa.gamint = 1.5  # !!!!!!!!!!!!!!!!!!!!!!!!
mfa.verbose = 1
mfa.wtype = 0

mfa.q = np.arange(-8, 9)

# get cumulants
mfa.analyze(data)
cp  = mfa.cumulants.log_cumulants
print("Noiseless cumulants: ")
print("c1 = ", cp[0])
print("c2 = ", cp[1])


#-------------------------------------------------------------------------------
# Run simulations
#-------------------------------------------------------------------------------
c1_list = np.zeros((len(p_list),noise_params['n_sigma']))
c2_list = np.zeros((len(p_list),noise_params['n_sigma']))
C1j_list = np.zeros((len(p_list),noise_params['n_sigma'], mfa.j2 - mfa.j1 + 1))
C2j_list = np.zeros((len(p_list),noise_params['n_sigma'], mfa.j2 - mfa.j1 + 1))

Dq_list = np.zeros((len(p_list),noise_params['n_sigma'], len(mfa.q)))
hq_list = np.zeros((len(p_list),noise_params['n_sigma'], len(mfa.q)))



for p_idx, p in enumerate(p_list):
    mfa.p = p
    for idx, ss2 in enumerate(sigma2):
        signal = data + np.sqrt(ss2)*noise    
        mfa.analyze(signal)
        cp  = mfa.cumulants.log_cumulants
        c1_list[p_idx, idx] = cp[0]
        c2_list[p_idx, idx] = cp[1]

        C1j_list[p_idx, idx, :] = mfa.cumulants.values[0, mfa.j1-1:mfa.j2]
        C2j_list[p_idx, idx, :] = mfa.cumulants.values[1, mfa.j1-1:mfa.j2]
        

        Dq_list[p_idx, idx, :] = mfa.spectrum.Dq
        hq_list[p_idx, idx, :] = mfa.spectrum.hq

        if idx % 15 == 0:
            print("--- simulation ", idx)



#-------------------------------------------------------------------------------
# Plots
#-------------------------------------------------------------------------------

sigmas_to_plot = np.percentile(sigma2, [0, 25, 50, 75, 100], interpolation = 'nearest')
sigma_indexes  = []
j_list = np.arange(mfa.j1, mfa.j2+1)


for ss2 in sigmas_to_plot:
    index = np.argsort( np.abs(sigma2 - ss2)  )[0]
    sigma_indexes.append(index)


# Log-cumulants

if NOISE_TYPE == 'white':
    title = 'White noise'

elif NOISE_TYPE == 'colored':
    title = 'Colored noise - beta = %0.2f'%noise_params['beta']

elif NOISE_TYPE == 'smooth':
    title = 'Smooth trend'

elif NOISE_TYPE == 'dirac':
    title = 'Dirac train'

plt.figure(1)
for p_idx, p in enumerate(p_list):
    plt.subplot(1, 2, 1)
    plt.plot(sigma2, c1_list[p_idx, :], 'o-' ,label = ('p = %0.1f'%p))
    plt.ylabel('$c_1$')
    plt.xlabel('$\sigma^2$')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    # plt.figure(2)
    plt.plot(sigma2, c2_list[p_idx, :], 'o-' ,label = ('p = %0.1f'%p))
    plt.ylabel('$c_2$')
    plt.xlabel('$\sigma^2$')
    plt.legend()
    plt.title(title)
    plt.grid(True)



# MF spectrum
plt.figure(3)

for p_idx, p in enumerate(p_list):

    # plt.figure(3 + p_idx)
    plt.subplot(1, len(p_list), p_idx+1)
    plt.title('p = %0.1f'%p_list[p_idx])

    for s_idx, sigma in enumerate(sigmas_to_plot):
        Dq = Dq_list[p_idx, sigma_indexes[s_idx], :]
        hq = hq_list[p_idx, sigma_indexes[s_idx], :]

        plt.plot(hq, Dq, 'o-',label = ('$\sigma^2$ = %0.2f'%sigma))

        plt.xlabel('$h$')
        plt.ylabel('$\mathcal{D}(h)$')

    plt.grid(True)
    plt.legend()


# # C1j
# for p_idx, p in enumerate(p_list):

#     plt.figure(3 + len(p_list) + p_idx)
#     plt.title('p = %0.1f'%p_list[p_idx])

#     for s_idx, sigma in enumerate(sigmas_to_plot):
#         C1j = C1j_list[p_idx, sigma_indexes[s_idx], :]

#         plt.plot(j_list, C1j, 'o-',label = ('$\sigma^2$ = %0.2f'%sigma))

#         plt.xlabel('$j$')
#         plt.ylabel('$C_1(j)$')

#     plt.grid(True)
#     plt.legend()

# # C2j
# for p_idx, p in enumerate(p_list):

#     plt.figure(3 + 2*len(p_list) + p_idx)
#     plt.title('p = %0.1f'%p_list[p_idx])

#     for s_idx, sigma in enumerate(sigmas_to_plot):
#         C2j = C2j_list[p_idx, sigma_indexes[s_idx], :]

#         plt.plot(j_list, C2j, 'o-',label = ('$\sigma^2$ = %0.2f'%sigma))

#         plt.xlabel('$j$')
#         plt.ylabel('$C_2(j)$')

#     plt.grid(True)
#     plt.legend()


plt.show()