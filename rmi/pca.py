# pca.py

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

class pca:
    '''
    Principal Component Analysis

    Simple implementation of the Principal Component Analysis (PCA). 
    Initialize the class with the 'training batch' and the number of components to consider:
    - x: training batch, with shape [N_large, dimension]
    - N_small: number of principal components to consider
    The class with automatically calculate the mean of the training batch and subtract it to the input.
    The projection matrix is saved in the variable w.
    Other useful variables are eigenvals, x_mean and N_small.

    You can apply the transformation with
    - transform(x)

    Another useful function is
    - plot_eigenvalues()
    '''

    def __init__(self, x, N_small):
        self.N_small = N_small

        self.x_mean = np.mean(x, axis=0)
        x_zero = x - self.x_mean
        x2 = np.dot(np.transpose(x_zero),x_zero)/(np.shape(x_zero)[0])
        x2_eigenvals,x2_eigenvects =np.linalg.eig(x2)
        x2_eigenvects=np.real(x2_eigenvects)
        x2_eigenvals=np.real(x2_eigenvals)

        idx = x2_eigenvals.argsort()[::-1]   
        x2_eigenvals = x2_eigenvals[idx]
        x2_eigenvects = x2_eigenvects[:,idx]

        self.eigenvals = x2_eigenvals
        self.w = x2_eigenvects[:,0: self.N_small].transpose()


    def transform(self,x):
        '''
        Applies the PCA transformation with N_small components. Requires:
        - x: batch to transform, with shape [N_large, dimension]
        '''
        return np.dot(self.w,x.transpose()).transpose()

    def plot_eigenvalues(self,output="none"):
        '''
        Plots all eigenvalues of the training sample, in decreasing order
        '''
        plt.figure()
        plt.title("Eigenvalues")
        plt.xlabel("#")
        plt.ylabel("Value")
        plt.scatter(range(self.eigenvals.shape[0]),self.eigenvals)
        if output != "none":
            plt.savefig("outputs/eigenvalues_" + output + ".png")
        else:
            plt.show()
