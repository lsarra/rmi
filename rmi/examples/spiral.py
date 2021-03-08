# spiral.py

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

class SpiralDistribution:
    """ Spiral-shaped distribution

    This class allows to sample from a 2d "spiral-shaped" distribution starting from a Gaussian distribution with given sx, sy, r = sigmax, sigmax, correlation coefficient as input. The twisting of the spiral is given by alpha.
    
    """

    def __init__(self, sx, sy, r, alpha):
        """Initialize the class 

        Args:
            sx ([type]): standard deviation of the 2d-Gaussian on the first component
            sy ([type]): standard deviation of the 2d-Gaussian on the second component
            r ([type]): correlation coefficient of the 2d-Gaussian distribution
            alpha ([type]): twist of the spiral (alpha=0 is a Gaussian with no twist)
        """

        self.sigmax = sx
        self.sigmay = sy
        self.rho = r
        self.alpha = alpha

        self.th = self.get_theta()[0]
        self.cov = self.get_covariance()
        self.inv_cov = np.linalg.inv(self.cov)

    def get_covariance(self):
        """ Covariance matrix of the initial Gaussian

        Returns:
            array_like: 2x2 covariance matrix of the associated Gaussian distribution
        """
        ''' 
        '''
        return [[self.sigmax**2, self.rho*self.sigmax*self.sigmay],
                [self.rho*self.sigmax*self.sigmay, self.sigmay**2]]

    def get_theta(self):
        """Direction of the Gaussian distribution

        Returns:
            list: the two sorted eigenvectors of the associated Gaussian distribution
        """

        cvmat = self.get_covariance()
        w, v = np.linalg.eig(cvmat)
        index = np.argmax(w)
        return [-np.arctan2(v[index][1], v[index][0]),
                -np.arctan2(v[1-index][1], v[1-index][0])]

    def spiralize_batch(self, x, alpha=None):
        """Deform a given batch of points
            Transforms a given batch into a spiral-shaped batch by applying the spiral transformation

                x' = x cos(alpha r) - y sin(alpha r)
                y' = x sin(alpha r) + y cos(alpha r)

            with r = sqrt(x^2 + y^2)

        Args:
            x (array_like): [N_samples, 2] array with the points to be transformed
            alpha (float, optional): Twist of the deformation. Defaults to class value.

        Returns:
            x_new (array_like): [N_samples, 2] array of transformed samples
        """
        '''

        '''
        if alpha is None:
            alpha = self.alpha
        r = np.sqrt(np.sum(x**2, -1))

        x_sp = x[:, 0]*np.cos(alpha*r) - x[:, 1]*np.sin(alpha*r)
        y_sp = x[:, 0]*np.sin(alpha*r) + x[:, 1]*np.cos(alpha*r)

        return np.array([x_sp, y_sp]).T

    def sample(self, N_samples):
        """Samples from the spiral distribution

        Args:
            N_samples (int): number of samples to generate

        Returns:
            array_like: [N_sample, 2] array of generated samples
        """

        return self.spiralize_batch(
            np.random.multivariate_normal([0, 0], self.cov, N_samples))

    def P_x(self, x):
        """Theoretical probability density at given points

        Calculates the theoretical probability density at the given points
        by performing the density transformation of the initial Gaussian distribution.

        Args:
            x (array_like): [N_points, 2] array of points where to calculate the probability density

        Returns:
            array_like: [N_points] theoretical P_x(x) of the given array of points
        """

        # Invert the transformation so that we can replace it into the distribution of a Gaussian
        x_g = self.spiralize_batch(x, -self.alpha)

        # The Jacobian of the transformation is 1. We only need to calculate
        # the Gaussian distribution in the old coordinates expressed as a function of the
        # transformed ones.
        return 1/np.sqrt((2*np.pi)**2*np.linalg.det(self.cov))*np.exp(-0.5*np.diag(np.dot(np.dot(x_g, self.inv_cov), x_g.T)))
