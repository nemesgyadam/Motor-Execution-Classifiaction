import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


def load_power(filename, cutoff=1000):
    powers = pd.read_csv(filename, header=None).to_numpy()
    powers = powers[:, 1:]
    outlier = np.where(powers > cutoff)
    # powers[outlier] = powers[np.clip(outlier - 1, 0, None)]
    powers[outlier] = 0
    return powers


def rm_digits(s):
    return ''.join([c for c in s if not c.isdigit()])


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
    
freq_incr = 250 / (2 * 250)
freqs = np.arange(0, 500) * freq_incr

delta_idx = np.where((freqs >= 1) & (freqs < 4))[0]
theta_idx = np.where((freqs >= 4) & (freqs < 7.5))[0]
alpha1_idx = np.where((freqs >= 7.5) & (freqs < 10))[0]
alpha2_idx = np.where((freqs >= 10) & (freqs < 13))[0]
alpha_idx = np.where((freqs >= 7.5) & (freqs < 13))[0]
smr_idx = np.where((freqs >= 13) & (freqs < 15))[0]
beta_idx = np.where((freqs >= 15) & (freqs < 30))[0]
beta1_idx = np.where((freqs >= 15) & (freqs < 20))[0]
beta2_idx = np.where((freqs >= 20) & (freqs < 30))[0]
gamma_idx = np.where((freqs >= 30) & (freqs < 45))[0]

# band_idx = [theta_idx, alpha_idx, alpha1_idx, alpha2_idx, smr_idx, beta1_idx, beta2_idx, gamma_idx]
# band_names = ['theta_idx', 'alpha_idx', 'alpha1_idx', 'alpha2_idx', 'smr_idx', 'beta1_idx', 'beta2_idx', 'gamma_idx']

band_idx = [delta_idx, theta_idx, alpha_idx, alpha1_idx, alpha2_idx, beta_idx, beta1_idx, beta2_idx, gamma_idx]
band_names = ['delta_idx', 'theta_idx', 'alpha_idx', 'alpha1_idx', 'alpha2_idx', 'beta_idx', 'beta1_idx', 'beta2_idx', 'gamma_idx']

# band_idx = [np.where((freqs >= s) & (freqs < e))[0] for s in range(7, 11) for e in range(s + 2, 13)]
# band_names = [f'{s}-{e}' for s in range(7, 11) for e in range(s + 2, 13)]

# names = ['powerz_arithmetic_meditation', 'powerz_arithmetic_viktor',
#          'powerz_eye_closed_arithmetic', 'powerz_eye_open_nothing']

names = ['powerz_gyozo_eye_closed_arithmetic', 'powerz_gyozo_eye_closed_meditation',
         'powerz_gyozo_eye_closed_nothing', 'powerz_gyozo_feedback_eye_closed_meditation']
# names = ['powerz_reka_eye_closed_arithmetic', 'powerz_reka_eye_closed_meditation',
#          'powerz_reka_eye_closed_nothing', 'powerz_reka_feedback_eye_closed_meditation_real_feedback']

data_path = 'c:/wut/asura/meditation'
files = [sorted(list(glob(f'{data_path}/{name}*.csv'))) for name in names]
powerz = [np.stack(list(map(load_power, fs))) for fs in files]
bandz = [[p[:, 0, idx].mean(axis=-1) for p in powerz] for idx in band_idx]  # select first channel (frontal)
colors = ['blue', 'orange', 'red', 'green', 'black']

# plt.figure()
# plt.plot(freqs, powerz[0][100, 0, :])
# plt.figure()
# plt.plot(freqs, powerz[1][150, 0, :])
# plt.figure()
# plt.hist(powerz[1].reshape(-1))
# plt.figure()
# plt.plot(freqs, powerz[2][10, 0, :])
# plt.show()

#fig, axes = plt.subplots(2, len(band_names) // 2)
#axes = axes.reshape(-1)
for j, (bz, band_name) in enumerate(zip(bandz, band_names)):
    fig, axes = plt.subplots(ncols=2)
    bzall = np.concatenate(bz)

    for i, name in enumerate(names):
        b = (bz[i] - bzall.mean()) / bzall.std()  # TODO
        # b = bz[i]
        bt = np.linspace(0, 100, b.shape[0])
        bhat = savitzky_golay(b, 25, 3)  # window size 13, polynomial order 3
        # bhat = b  # TODO

        # for i in range(bandz.shape[0]):
        #     axes[0, j].plot(bandz[i], label=names[i])
        
        axes[0].plot(bt, bhat, label=name, alpha=.5, color=colors[i])
        axes[1].hist(bhat, bins=30, alpha=.4, color=colors[i], label=name, density=True)
        if i == len(names) - 1:
            fig.legend(loc='lower center',  ncol=2)
        plt.tight_layout()
    fig.suptitle(band_name)
    
    # plt.legend()
    plt.tight_layout()
    
plt.show()

print('i')
