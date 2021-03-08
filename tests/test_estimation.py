import os  # NOQA
import sys  # NOQA
sys.path.insert(0, os.path.abspath('..'))  # NOQA

import unittest
import numpy as np
import rmi.estimation as inf


class TestProbability(unittest.TestCase):

    def test_density_gauss_1d(self):
        # Arrange
        mu = 0
        sigma = 1
        n_bins = 180
        N_points = int(10e5)
        samples = np.random.randn(N_points)*sigma + mu

        p_min = np.min(samples)
        p_max = np.max(samples)
        points_linsp = np.linspace(p_min, p_max, n_bins)

        # Act
        dx_expected = points_linsp[1]-points_linsp[0]
        px_expected = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*(points_linsp-mu)**2/sigma**2)

        px, dx = inf.produce_P(samples, n_bins=n_bins)

        # Assert
        np.testing.assert_almost_equal(dx, dx_expected, decimal=3)
        np.testing.assert_almost_equal(px, px_expected, decimal=2)

    def test_density_uniform_1d(self):
        # Arrange
        u_min = -1
        u_max = 3
        n_bins = 180
        N_points = int(10e5)
        samples = np.random.uniform(u_min, u_max, N_points)

        points_linsp = np.linspace(u_min, u_max, n_bins)

        # Act
        dx_expected = points_linsp[1]-points_linsp[0]
        px_expected = np.full(n_bins, 1/n_bins/dx_expected)

        px, dx = inf.produce_P(samples, n_bins=n_bins)

        # Assert
        np.testing.assert_almost_equal(dx, dx_expected, decimal=3)
        np.testing.assert_almost_equal(px, px_expected, decimal=2)

    def test_density_gauss_2d(self):
        # Arrange
        mu = np.array([0, 1])
        sigma = np.array([[1, 0.5], [0.5, 2]])
        det_sigma = np.linalg.det(sigma)

        n_bins = 100
        N_points = int(10e6)
        samples = np.random.multivariate_normal(mu, sigma, N_points)

        p_min = np.min(samples, 0)
        p_max = np.max(samples, 0)
        points_linspX = np.linspace(p_min[0], p_max[0], n_bins)
        points_linspY = np.linspace(p_min[1], p_max[1], n_bins)

        mesh_x, mesh_y = np.meshgrid(points_linspX, points_linspY)
        point_x = mesh_x.flatten()
        point_y = mesh_y.flatten()
        points = np.array([point_x, point_y]).T

        # Act
        dx_expected = (p_max - p_min)/n_bins
        px_expected = 1/np.sqrt((2*np.pi)**2*det_sigma) * \
            np.exp(-0.5*np.diag(np.dot((points-mu), np.dot((points-mu), np.linalg.inv(sigma)).T)))

        px_expected = px_expected.reshape([n_bins, n_bins]).T
        px, dx = inf.produce_P(samples, n_bins=n_bins)

        # Assert
        np.testing.assert_almost_equal(dx, dx_expected, decimal=3)
        np.testing.assert_almost_equal(px, px_expected, decimal=2)

    def test_density_uniform_2d(self):
        # Arrange
        u_min = np.array([-1, 2])
        u_max = np.array([2, 4])

        n_bins = 100
        N_points = int(10e7)
        samples = np.random.uniform(0, 1, [N_points, 2])*(u_max-u_min) + u_min

        # Act
        dx_expected = (u_max - u_min)/n_bins
        px_expected = np.full([n_bins, n_bins], 1/n_bins**2/np.prod(dx_expected))
        px, dx = inf.produce_P(samples, n_bins=n_bins)

        # Assert
        np.testing.assert_almost_equal(dx, dx_expected, decimal=3)
        np.testing.assert_almost_equal(px, px_expected, decimal=2)


class TestEntropy(unittest.TestCase):
    def test_entropy_gauss_1d(self):
        # Arrange
        mu = 0
        sigma = 1
        N_points = int(10e6)
        samples = np.random.randn(N_points)*sigma + mu

        # Act
        entropy_expected = inf.H_gaussian_1d_theoretical(mu, sigma)
        entropy = inf.Entropy(*inf.produce_P(samples))

        # Assert
        np.testing.assert_almost_equal(entropy, entropy_expected, decimal=3)

    def test_entropy_uniform_1d(self):
        # Arrange
        u_min = -1
        u_max = 3
        N_points = int(10e6)
        samples = np.random.uniform(u_min, u_max, N_points)

        # Act
        entropy_expected = inf.H_uniform_1d_theoretical(u_min, u_max)
        entropy = inf.Entropy(*inf.produce_P(samples))

        # Assert
        np.testing.assert_almost_equal(entropy, entropy_expected, decimal=3)

    def test_entropy_gauss_2d(self):
        # Arrange
        mu = np.array([0, 1])
        sigma = np.array([[1, 0.5], [0.5, 2]])

        N_points = int(10e6)
        samples = np.random.multivariate_normal(mu, sigma, N_points)

        # Act
        entropy_expected = inf.H_gaussian_2d_theoretical(mu, sigma)
        entropy = inf.Entropy(*inf.produce_P(samples))

        # Assert
        np.testing.assert_almost_equal(entropy, entropy_expected, decimal=3)

    def test_entropy_uniform_2d(self):
        # Arrange
        u_min = np.array([-1, 2])
        u_max = np.array([2, 4])

        N_points = int(10e7)
        samples = np.random.uniform(0, 1, [N_points, 2])*(u_max-u_min) + u_min

        # Act
        entropy_expected = inf.H_uniform_2d_theoretical(u_min, u_max)
        entropy = inf.Entropy(*inf.produce_P(samples))

        # Assert
        np.testing.assert_almost_equal(entropy, entropy_expected, decimal=3)


class TestRenormalizedMutualInformation(unittest.TestCase):

    def test_regterm_linear(self):
        # Arrange
        N_points = int(10e3)
        th = np.random.uniform(0, 2*np.pi, N_points)
        grad = np.array([np.cos(th), np.sin(th)]).T[:, None, :]

        # Act
        reg = inf.RegTerm(grad)
        reg_expected = 0

        # Assert
        np.testing.assert_almost_equal(reg, reg_expected, decimal=3)

    def test_repar_inv_cubic(self):
        # Arrange
        mu = 0
        sigma = 1
        N_points = int(10e5)
        samples = np.random.randn(N_points)*sigma + mu

        feature = samples**3 + 5*samples
        gradient = 3*samples**2 + 5

        # Act
        rmi = inf.RenormalizedMutualInformation(feature[:, None], gradient[:, None, None])
        rmi_expected = inf.H_gaussian_1d_theoretical(mu, sigma)

        # Assert
        np.testing.assert_almost_equal(rmi, rmi_expected, decimal=2)
