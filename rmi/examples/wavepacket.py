# wavepacket.py

# Copyright (C) 2021

# Code by Leopoldo Sarra and Florian Marquardt
# Max Planck Institute for the Science of Light, Erlangen, Germany
# http://www.mpl.mpg.de


# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# If you find this code useful in your work, please cite our article
# "Renormalized Mutual Information for Artificial Scientific Discovery", Leopoldo Sarra, Andrea Aiello, Florian Marquardt, arXiv:2005.01912

# available on

# https://arxiv.org/abs/2005.01912

# ------------------------------------------

"""
One dimensional field with some background noise and a fixed-shaped wave packet (Gaussian)

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def produce_Wave_Packet(
        n_pixels=100,
        n_samples=10000,
        width=3.0,
        noise=0.0,
        noisy_amplitude=None,
        pos_range=None,
        colored_noise=None,
        pos_avg_span=False,
        random_strength=False):
    """Create samples of a one dimensional wave packet over a fluctuating field:

    Creates samples by adding
    - background fluctuating field (random Gaussian around zero, std=noise)
    - packet of fixed shape (in this case, bell-shaped, centered at a random position) 

    Args:
        n_pixels (int, optional): [description]. Defaults to 100.
        n_samples (int, optional): [description]. Defaults to 10000.
        width (float, optional): shape of the packet e^(-((j-jbar)/width)^2). Defaults to 3.0.
        noise (float, optional): std of the background noise. Defaults to 0.0.
        noisy_amplitude (float, optional):  dA: the wave packet has amplitude (1+dA), if not given dA=0. Defaults to None.
        pos_range (list, optional): [jmin, jmax], possible position of the center of the wave packet (uniformly distributed in this interval). If n_pixels=100, we suggest [30,70]. This is to avoid possible boundary effects (the wavepacket is cut). Defaults to None.
        colored_noise (float, optional): use colored noise instead of uncorrelated. Defaults to None.
        pos_avg_span (bool, optional): instead of placing the center of the packet randomly (uniformly), the position is assigned sequentially. Defaults to False.
        random_strength (bool, optional): random noise std (instead of fixing it to std=noise). noise argument is ignored. Defaults to False.

    Returns:
        samples (array_like): [n_samples, n_pixels] generated samples
        pos (array_like): [n_samples] center of the wave-packet 

    Example: ::

            produce_Wave_Packet(n_pixels=100,
                                n_samples=200,
                                width=9.0,
                                noise=0.8,
                                pos_range=[30,70])

    """

    x = np.array(range(n_pixels))

    if random_strength:
        noise = np.random.uniform(0, 0.3, size=(n_samples, 1))

    #####################
    # Background noise
    if colored_noise is None:
        background_noise = noise*np.random.randn(n_samples, n_pixels)
    else:
        k = np.fft.fftfreq(n_pixels)
        A_k = (np.random.randn(n_samples, n_pixels)+1j *
               np.random.randn(n_samples, n_pixels))/(1+(colored_noise*k)**2)
        # not clear the meaning of xi = colored_noise
        background_noise = noise*np.real(np.fft.fft(A_k, axis=1))

    #####################
    # Wave Packet

    # wp center position range
    if pos_range is None:
        pos_range = [0, n_pixels]
    pos_width = pos_range[1] - pos_range[0]

    if pos_avg_span is False:
        pos = pos_range[0] + np.random.random(n_samples)*pos_width
    else:
        pos = pos_range[0] + np.linspace(0., 1., n_samples)*pos_width

    # Amplitude fluctuation
    if noisy_amplitude is None:
        amp = 1.0
    else:
        # shape [N_samples, N_pixels]
        amp = 1 + noisy_amplitude*np.random.randn(n_samples)[:, np.newaxis]

    wave_packet = amp * \
        np.exp(- (x[np.newaxis, :] - pos[:, np.newaxis])**2/width**2)

    #####################
    # Final Samples
    samples = background_noise + wave_packet

    # # Save for debug (can be safely removed)
    # global last_background, last_wp, last_pos
    # last_background = background_noise
    # last_wp = wave_packet
    # last_pos = pos

    return samples, pos


def plot(samples, N_samples=None, save_path=None):
    """Plots the first N_samples. 
    
    Color represents the intensity of the field, X is the coordinate of the field (i.e. j) and Y represents the number of samples.

    Args:
        samples (array_like): [n_samples, n_pixels] samples to plot.
        N_samples (int, optional): How many samples to plot. Defaults to at most 100.
        save_path (str, optional): Path where to save the plot. Defaults to None.
    """
    if N_samples is None:
        N_samples = np.minimum(len(samples), 100)

    if save_path is None:
        plt.figure(figsize=[10, 10])
        plt.xlabel("x")
        plt.ylabel("# sample")
    else:
        font = {'family': 'DejaVu Serif',
                'weight': 'regular',
                'size': 22}
        mpl.rc('font', **font)
        plt.figure(figsize=[5, 5], dpi=300)
        plt.axis("off")

    plt.imshow(samples[0:N_samples, :], cmap=plt.cm.viridis)
    if save_path is not None:
        plt.savefig(save_path + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
    mpl.style.use('default')


def plot_sorted_feature(samples, feature, save_path=None):
    """Plots the first 100 wave packets, sorted according to feature value

    Selects the first 100 wave-packets. 
    Orders them according to the value of the associated feature.
    Plots the wave-packets one by row (color encodes the value of the field.)

    Args:
        samples (array_like): [n_samples, n_pixels]
        feature (array_like): [n_samples]
        last_pos: should not be required
        save_path (str, optional): Path where to save the plot. Defaults to None.
    """
    feature_samples = feature[:100, 0]
    small_samples = samples[:100]
    new_smpl = small_samples[np.argsort(feature_samples)]
    plot(new_smpl, save_path=save_path)

