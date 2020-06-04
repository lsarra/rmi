'''
 features.py

Copyright (C) 2020

Code by Leopoldo Sarra and Florian Marquardt
Max Planck Institute for the Science of Light, Erlangen, Germany
http://www.mpl.mpg.de

This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

If you find this code useful in your work, please cite our article
"Renormalized Mutual Information for Artificial Scientific Discovery", Leopoldo Sarra, Andrea Aiello, Florian Marquardt, arXiv:2005.01912

available on

https://arxiv.org/abs/2005.01912

------------------------------------------

 The class Features lists some of the most common features that can be studied. Each of the functions returns the feature and its gradient calculated on the given Samples

 Usage:
    - from features import *
    - f = Features()
'''

import numpy as np
import matplotlib.pyplot as plt

# Some common (1D) features
# Output should be [Feature, Grad]


class Features:
    '''
    This class implements the most common one-dimensional features and a function to join two 1d features into a 2d feature (JoinTwo). Each function
    - takes as input the Samples array, with required shape [N_samples, N_x]
    - outputs the feature [N_samples, N_y] and its gradient [N_samples, N_y, N_x]
    Additional arguments are provided after the first argument. Please use lambda functions to specify additional arguments when giving a feature function to JoinTwo or Information.print_feature_batch()
    '''
    epsilon = 1e-9

    def pca(self, samples, N_small=1, plot=False):
        ''' Returns the first N_small largest eigenvectors given by the PCA. This function uses the PCA library (pca.py), which should be placed in the same folder.
        If plot, shows the plot of the eigenvalues, ordered by size.
        '''
        from utils.pca import pca
        mypca = pca(samples, N_small)

        Feature = mypca.transform(samples)
        Grad = np.repeat(mypca.w[np.newaxis, :N_small],
                         samples.shape[0], axis=0)
        if plot:
            mypca.plot_eigenvalues()
        return Feature, Grad

    def cm(self, samples, N_particles=None):
        '''
         Get the center-of-mass of the first N_particles: F=(x1+x2+...)/N
        '''
        Nx = N_particles
        if Nx == None:
            Nx = samples.shape[-1]

        Feature = np.sum(samples[:, :Nx], axis=1)/Nx
        Grad = np.full(np.shape(samples), 1.0/Nx)
        Grad[:, Nx:] = 0

        return Feature, Grad

    def sum_x_j(self, samples):
        '''
         Get F(x1,x2,..)= (1*x1**2+2*x2**2+...)= sum j x_j
        '''
        Nx = samples.shape[-1]
        js = np.arange(Nx)
        Feature = np.sum(js[np.newaxis, :]*samples, axis=1)
        Grad = np.repeat(js[np.newaxis, :], samples.shape[0], axis=0)
        return Feature, Grad

    def sum_x(self, samples):
        '''
        Get F=(x1+x2+...)
        '''
        Nx = samples.shape[-1]

        Feature = np.sum(samples[:, :Nx], axis=1)
        Grad = np.full(np.shape(samples), 1.0)
        return Feature, Grad

    # <x|j|x>
    def qm_j(self, samples):
        '''
         Get quantum average position: F= sum{j x_j^2 }/sum{x_j^2} = <X|J|X>
        '''
        Nx = samples.shape[-1]
        js = np.arange(Nx)
        normalization = np.sum(samples**2, axis=1)
        Feature = np.sum(js[np.newaxis, :]*samples**2, axis=1)/normalization

        Grad = 2*samples*(js[np.newaxis, :] - Feature[:,
                                                      np.newaxis])/normalization[:, np.newaxis]
        return Feature, Grad

    def sum_x2_j(self, samples):
        '''
         Get F(x1,x2,..)= (1*x1**2+2*x2**2+...) = sum j x_j^2
        '''
        Nx = samples.shape[-1]
        js = np.arange(Nx)
        Feature = np.sum(js[np.newaxis, :]*samples**2, axis=1)
        Grad = 2*samples*js[np.newaxis, :]
        return Feature, Grad

    def sum_xalpha_j(self, samples, alpha):
        '''
        Get F(x1,x2,..)= (1*x1**alpha+2*x2**alpha+...) = sum j x_j^alpha
        '''
        Nx = samples.shape[-1]
        js = np.arange(Nx)
        Feature = np.sum(js[np.newaxis, :]*samples**alpha, axis=1)
        Grad = alpha*samples**(alpha-1)*js[np.newaxis, :]
        return Feature, Grad

    def sqrt_x2(self, samples):
        '''
        Get F = sqrt{<x_j^2>}
        '''
        Nx = samples.shape[-1]
        Feature = np.sqrt((1/Nx)*np.sum(samples**2, axis=1))
        Grad = samples/Nx/Feature.reshape([-1, 1])
        return Feature, Grad

    def x2(self, samples):
        '''
        Get F = <x_j^2>
        '''
        Nx = samples.shape[-1]
        Feature = np.sum(samples**2, axis=1)/Nx
        Grad = 2*samples/Nx
        return Feature, Grad

    def var(self, samples):
        '''
        Variance of the samples
        '''
        Nx = samples.shape[-1]
        x_cm, _ = self.cm(samples)

        Feature = 1./Nx * np.sum((samples-x_cm[:, np.newaxis])**2, axis=1)
        Grad = 2./Nx * (samples - x_cm[:, np.newaxis])
        return Feature, Grad

    def sqrt_var(self, samples):
        '''
         Standard deviation of the samples
        '''
        Nx = samples.shape[-1]
        x_cm, _ = self.cm(samples)

        Feature = np.sqrt(
            1./Nx * np.sum((samples-x_cm[:, np.newaxis])**2, axis=1))
        Grad = 1./Nx * (samples - x_cm[:, np.newaxis])/Feature[:, np.newaxis]
        return Feature, Grad

    def linear(self, x, th):
        '''
        2d input: linear increasing in the direction given by angle th.
        '''
        Feature = x[:, 0]*np.cos(th) + x[:, 1]*np.sin(th)
        Grad1 = np.full(np.shape(x)[0], np.cos(th))
        Grad2 = np.full(np.shape(x)[0], np.sin(th))
        return Feature, np.array([Grad1, Grad2]).T

    def linear3d(self, x, th, phi):
        '''
        3d input: linear increasing in the direction given by angles th and phi.
        '''
        n = np.array([np.sin(phi)*np.cos(th), np.sin(phi)
                      * np.sin(th), np.cos(phi)])
        Feature = np.dot(x, n)
        Grad1 = np.full(np.shape(x)[0], n[0])
        Grad2 = np.full(np.shape(x)[0], n[1])
        Grad3 = np.full(np.shape(x)[0], n[2])
        return Feature, np.array([Grad1, Grad2, Grad3]).T

    def spiral(self, x, th, alpha):
        '''
        2d input: spiral features in direction th, with twisting alpha.
        '''
        r = np.sqrt(np.sum(x**2, -1))
        ang = th + alpha*r
        c = np.cos(ang)
        s = np.sin(ang)

        Feature = x[:, 0]*c + x[:, 1]*s

        arg = alpha/(r+self.epsilon)*(-x[:, 0]*s + x[:, 1]*c)
        Grad1 = c + x[:, 0]*arg
        Grad2 = s + x[:, 1]*arg
        return Feature, np.array([Grad1, Grad2]).T

    def sum_x_j2(self, samples):
        '''
        Get F= sum{j^2 x}
        '''
        Nx = samples.shape[-1]
        js = np.arange(Nx)
        Feature = np.sum(js[np.newaxis, :]**2*samples, axis=1)
        Grad = np.repeat(js[np.newaxis, :]**2, samples.shape[0], axis=0)
        return Feature, Grad

    def qm_j_beta(self, samples, beta):
        '''
         Get a generalized quantum average position: F= <j x_j^2>/<x_j^beta> = <X|J|X>
        '''
        Nx = samples.shape[-1]
        js = np.arange(Nx)
        normalization = np.sum(samples**beta, axis=1)
        Feature = np.sum(js[np.newaxis, :]*samples**2, axis=1)/normalization

        Grad = 2*(samples*js[np.newaxis, :] - beta*Feature[:, np.newaxis]
                  * samples**(beta-1))/normalization[:, np.newaxis]
        return Feature, Grad

    def joinTwo(self, Samples, f1_f, f2_f):
        '''
        Joins two given 1d features into a single 2d feature. Please provide the two feature functions as arguments (not yet calculated on data). Use lambda functions to provide additional arguments.
        '''
        f1 = f1_f(Samples)
        f2 = f2_f(Samples)
        Feature = np.array([f1[0].flatten(), f2[0].flatten()]).T
        Grad = np.array([f1[1], f2[1]]).swapaxes(0, 1)
        return Feature, Grad
