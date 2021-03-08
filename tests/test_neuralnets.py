import os  # NOQA
import sys  # NOQA
sys.path.insert(0, os.path.abspath('..'))  # NOQA

import unittest
import numpy as np
import rmi.estimation as inf
import rmi.neuralnets as nn


class TestProbability(unittest.TestCase):

    def test_density_gauss_1d(self):
        # Arrange
        mu = 0
        sigma = 1
        n_bins = 180
        N_points = int(10e5)
        samples = np.random.randn(N_points, 1)*sigma + mu

        p_min = np.min(samples)
        p_max = np.max(samples)
        delta = 3 * (p_max-p_min)/n_bins
        points_linsp = np.linspace(p_min-delta, p_max+delta, n_bins)

        net = nn.RMIOptimizer(H_nbins=n_bins)

        # Act
        dx_expected = points_linsp[1]-points_linsp[0]
        px_expected = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*(points_linsp-mu)**2/sigma**2)

        tf_y = nn.tf.convert_to_tensor(samples.T, nn.K.backend.floatx())
        px, dx = net.tf_calcProbabilityDistribution(tf_y)
        px = px.numpy()
        dx = dx.numpy()

        # Assert
        np.testing.assert_almost_equal(px, px_expected, decimal=2)

    def test_density_gauss_2d(self):
        # Arrange
        mu = np.array([0, 1])
        sigma = np.array([[1, 0.5], [0.5, 2]])
        det_sigma = np.linalg.det(sigma)

        n_bins = 100
        N_points = int(10e4)
        samples = np.random.multivariate_normal(mu, sigma, N_points)

        p_min = np.min(samples, 0)
        p_max = np.max(samples, 0)
        delta = 3 * (p_max-p_min)/n_bins

        points_linspX = np.linspace(p_min[0]-delta[0], p_max[0]+delta[0], n_bins)
        points_linspY = np.linspace(p_min[1]-delta[0], p_max[1]+delta[0], n_bins)

        mesh_x, mesh_y = np.meshgrid(points_linspX, points_linspY)
        point_x = mesh_x.flatten()
        point_y = mesh_y.flatten()
        points = np.array([point_x, point_y]).T

        net = nn.RMIOptimizer(H_nbins=n_bins)

        # Act
        dx_expected = (p_max - p_min)/n_bins
        px_expected = 1/np.sqrt((2*np.pi)**2*det_sigma) * \
            np.exp(-0.5*np.diag(np.dot((points-mu), np.dot((points-mu), np.linalg.inv(sigma)).T)))

        px_expected = px_expected.reshape([n_bins, n_bins]).T

        tf_y = nn.tf.convert_to_tensor(samples.T, nn.K.backend.floatx())

        px, dx = net.tf_calcProbabilityDistribution(tf_y)
        px = px.numpy()
        dx = dx.numpy()

        # Assert
        np.testing.assert_almost_equal(dx, dx_expected, decimal=2)
        np.testing.assert_almost_equal(px, px_expected, decimal=2)
