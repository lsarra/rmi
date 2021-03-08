# features.py

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


import numpy as np
import matplotlib.pyplot as plt
from .pca import pca as PCA

'''
Some common features that can be studied. E
ach of the functions returns the feature and its gradient calculated on the given samples

This class implements the most common one-dimensional features and a function to join two 1d features into a 2d feature (JoinTwo). 
Each function
- takes as input the samples array, with required shape [N_samples, N_x]
- outputs the feature [N_samples, N_y] and its gradient [N_samples, N_y, N_x]
Additional arguments are provided after the first argument. 
Please use lambda functions to specify additional arguments when giving a feature function to JoinTwo or estimation.print_feature_batch()
'''


def pca(samples, N_small=1, plot=False):
    """Returns the first N_small largest eigenvectors given by the PCA.
    This function uses the PCA module (pca.py)

    Args:
        samples (array_like): [N_samples, N_x] array of samples
        N_small (int, optional): Number of components. Defaults to 1.
        plot (bool, optional): Shows the plot of the eigenvalues, ordered by size. Defaults to False.

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """

    mypca = PCA(samples, N_small)

    Feature = mypca.transform(samples)
    Grad = np.repeat(mypca.w[np.newaxis, :N_small],
                     samples.shape[0], axis=0)
    if plot:
        mypca.plot_eigenvalues()
    return Feature, Grad


def cm(samples, N_particles=None):
    """Get the center-of-mass of the first N_particles

    F=(x1+x2+...)/N

    Args:
        samples (array_like): [N_samples, N_x] array of samples

        N_particles (int, optional): number of particles to consider. Defaults to None.

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = N_particles
    if Nx == None:
        Nx = samples.shape[-1]

    Feature = np.sum(samples[:, :Nx], axis=1)/Nx
    Grad = np.full(np.shape(samples), 1.0/Nx)
    Grad[:, Nx:] = 0

    return Feature, Grad


def sum_x_j(samples):
    """Get F(x1,x2,..)= (1*x1**2+2*x2**2+...)= sum j x_j

    Args:
        samples (array_like): [N_samples, N_x] array of samples

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = samples.shape[-1]
    js = np.arange(Nx)
    Feature = np.sum(js[np.newaxis, :]*samples, axis=1)
    Grad = np.repeat(js[np.newaxis, :], samples.shape[0], axis=0)
    return Feature, Grad


def sum_x(samples):
    """Get F=(x1+x2+...)

    Args:
        samples (array_like): [N_samples, N_x] array of samples

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = samples.shape[-1]

    Feature = np.sum(samples[:, :Nx], axis=1)
    Grad = np.full(np.shape(samples), 1.0)
    return Feature, Grad


def qm_j(samples):
    """Get quantum average position
        <x|j|x>

    F = sum{j x_j^2 }/sum{x_j^2} = <X|J|X>

    Args:
        samples (array_like): [N_samples, N_x] array of samples

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = samples.shape[-1]
    js = np.arange(Nx)
    normalization = np.sum(samples**2, axis=1)
    Feature = np.sum(js[np.newaxis, :]*samples**2, axis=1)/normalization

    Grad = 2*samples*(js[np.newaxis, :] - Feature[:,
                                                  np.newaxis])/normalization[:, np.newaxis]
    return Feature, Grad


def sum_x2_j(samples):
    """Get F(x1,x2,..)= (1*x1**2+2*x2**2+...) = sum j x_j^2

    Args:
        samples (array_like): [N_samples, N_x] array of samples

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = samples.shape[-1]
    js = np.arange(Nx)
    Feature = np.sum(js[np.newaxis, :]*samples**2, axis=1)
    Grad = 2*samples*js[np.newaxis, :]
    return Feature, Grad


def sum_xalpha_j(samples, alpha):
    """Get F(x1,x2,..)= (1*x1**alpha+2*x2**alpha+...) = sum j x_j^alpha


    Args:
        samples (array_like): [N_samples, N_x] array of samples

        alpha (float): power of x

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    '''
    '''
    Nx = samples.shape[-1]
    js = np.arange(Nx)
    Feature = np.sum(js[np.newaxis, :]*samples**alpha, axis=1)
    Grad = alpha*samples**(alpha-1)*js[np.newaxis, :]
    return Feature, Grad


def sqrt_x2(samples):
    """Get F = sqrt{<x_j^2>}

    Args:
        samples (array_like): [N_samples, N_x] array of samples

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = samples.shape[-1]
    Feature = np.sqrt((1/Nx)*np.sum(samples**2, axis=1))
    Grad = samples/Nx/Feature.reshape([-1, 1])
    return Feature, Grad


def x2(samples):
    """Get F = <x_j^2>

    Args:
        samples (array_like): [N_samples, N_x] array of samples

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = samples.shape[-1]
    Feature = np.sum(samples**2, axis=1)/Nx
    Grad = 2*samples/Nx
    return Feature, Grad


def var(samples):
    """Variance of the samples

    Args:
        samples (array_like): [N_samples, N_x] array of samples


    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = samples.shape[-1]
    x_cm, _ = cm(samples)

    Feature = 1./Nx * np.sum((samples-x_cm[:, np.newaxis])**2, axis=1)
    Grad = 2./Nx * (samples - x_cm[:, np.newaxis])
    return Feature, Grad


def sqrt_var(samples):
    """Standard deviation of the samples

    Args:
        samples (array_like): [N_samples, N_x] array of samples

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = samples.shape[-1]
    x_cm, _ = cm(samples)

    Feature = np.sqrt(
        1./Nx * np.sum((samples-x_cm[:, np.newaxis])**2, axis=1))
    Grad = 1./Nx * (samples - x_cm[:, np.newaxis])/Feature[:, np.newaxis]
    return Feature, Grad


def linear(x, th):
    """ linear increasing in the direction given by angle th.

    Args:
        x (array_like): [N_samples, 2] array of samples
        th (float): direction of the feature in which it increases

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """

    Feature = x[:, 0]*np.cos(th) + x[:, 1]*np.sin(th)
    Grad1 = np.full(np.shape(x)[0], np.cos(th))
    Grad2 = np.full(np.shape(x)[0], np.sin(th))
    return Feature, np.array([Grad1, Grad2]).T


def linear3d(x, th, phi):
    """ linear increasing in the direction given by angles th and phi.

    Args:
        x (array_like): [N_samples, 3] array of samples

        th ([type]): [description]
        phi ([type]): [description]

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    n = np.array([np.sin(phi)*np.cos(th), np.sin(phi)
                  * np.sin(th), np.cos(phi)])
    Feature = np.dot(x, n)
    Grad1 = np.full(np.shape(x)[0], n[0])
    Grad2 = np.full(np.shape(x)[0], n[1])
    Grad3 = np.full(np.shape(x)[0], n[2])
    return Feature, np.array([Grad1, Grad2, Grad3]).T


def spiral(x, th, alpha):
    """Spiral shaped feature

    Please NOTE: this is **not** the optimal feature for a spiral-shaped distribution!

    Args:
        x (array_like): [N_samples, 2] array of samples

        th (float): rotation
        alpha (float): twist

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    r = np.sqrt(np.sum(x**2, -1))
    ang = th + alpha*r
    c = np.cos(ang)
    s = np.sin(ang)

    Feature = x[:, 0]*c + x[:, 1]*s

    arg = alpha/(r+1e-9)*(-x[:, 0]*s + x[:, 1]*c)
    Grad1 = c + x[:, 0]*arg
    Grad2 = s + x[:, 1]*arg
    return Feature, np.array([Grad1, Grad2]).T


def sum_x_j2(samples):
    """ Get F= sum{j^2 x}

    Args:
        samples (array_like): [N_samples, N_x] array of samples

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = samples.shape[-1]
    js = np.arange(Nx)
    Feature = np.sum(js[np.newaxis, :]**2*samples, axis=1)
    Grad = np.repeat(js[np.newaxis, :]**2, samples.shape[0], axis=0)
    return Feature, Grad


def qm_j_beta(samples, beta):
    """ Get a generalized quantum average position: 

    F= <j x_j^2>/<x_j^beta> = <X|J|X>

    Args:
        samples (array_like): [N_samples, N_x] array of samples
        beta ([type]): power of normalization (just to explore)

    Returns:
        feature (array_like): [N_samples, 1] feature
        grad_feature (array_like): [N_samples, 1, N_x] gradient of the feature
    """
    Nx = samples.shape[-1]
    js = np.arange(Nx)
    normalization = np.sum(samples**beta, axis=1)
    Feature = np.sum(js[np.newaxis, :]*samples**2, axis=1)/normalization

    Grad = 2*(samples*js[np.newaxis, :] - beta*Feature[:, np.newaxis]
              * samples**(beta-1))/normalization[:, np.newaxis]
    return Feature, Grad


def joinTwo(Samples, f1_f, f2_f):
    """    Joins two given 1d features into a single 2d feature. 

    Please provide the two feature functions as arguments (not yet calculated on data). 
    Use lambda functions to provide additional arguments.

    Args:
        samples (array_like): [N_samples, N_x] array of samples
        f1_f (function): first feature
        f2_f (function): second feature

    Returns:
        feature (array_like): [N_samples, 2] feature
        grad_feature (array_like): [N_samples, 2, N_x] gradient of the feature
    """

    f1 = f1_f(Samples)
    f2 = f2_f(Samples)
    Feature = np.array([f1[0].flatten(), f2[0].flatten()]).T
    Grad = np.array([f1[1], f2[1]]).swapaxes(0, 1)
    return Feature, Grad
