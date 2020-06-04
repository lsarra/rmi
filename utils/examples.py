'''
examples.py

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


Simple physical system to analyze with Renormalized Mutual Information. 
Usage:
- Initialize the class with required parameters
- Each class has a function called Samples(N_samples) which returns N_samples with shape [N_samples, N_x]

This package implements the following systems:
- Random Gaussian
- Spiral Distribution
- Wave Packet
- Simulator (particles in one dimensional potential)
- Random Molecules (one dimensional gas)

'''


import numpy as np
import matplotlib.pyplot as plt


class RandomGaussian:
    '''
    This class initializes a gaussian distribution in a random way and samples points from it.
    '''

    def __init__(self, N):
        self.N = N
        self.mu0 = np.zeros(N)
        self.cov0 = self.get_covariance()

    def get_covariance(self):
        '''
        Generates a new random convariance matrix
        '''
        correlations = np.random.randn(self.N)
        noise = np.random.uniform(-0.5, 0.5, size=self.N)
        noise = noise[:, np.newaxis]*noise[np.newaxis, :]
        np.fill_diagonal(noise, 1)
        correlated = correlations[:, np.newaxis] * \
            correlations[np.newaxis, :]*noise
        return correlated

    def sample(self, N_samples):
        ''' Returns N_samples from the distribution, with shape [N_samples, N]'''
        return np.random.multivariate_normal(self.mu0, self.cov0, N_samples)


class SpiralDistribution:
    '''
    This class initializes a 2d "spiral-shaped" distribution starting from a gaussian distribution with given sx, sy, r = sigmax, sigmax, correlation coefficient as input.
    The twisting of the spiral is given by alpha
    '''

    def __init__(self, sx, sy, r, alpha):
        ''' Initialize the class by giving the sigmas and the correlation coefficient of the gaussian distribution, from which to create a spiral-shaped distribution. Alpha is the twist of the spiral
        '''
        self.sigmax = sx
        self.sigmay = sy
        self.rho = r
        self.alpha = alpha

        self.th = self.get_theta()[0]
        self.cov = self.get_covariance()
        self.inv_cov = np.linalg.inv(self.cov)

    def get_covariance(self):
        ''' Returns the covariance matrix of the associated gaussian distribution
        '''
        return [[self.sigmax**2, self.rho*self.sigmax*self.sigmay],
                [self.rho*self.sigmax*self.sigmay, self.sigmay**2]]

    def get_theta(self):
        '''
        Returns the directions of the eigenvectors of the associated gaussian distribution
        '''
        cvmat = self.get_covariance()
        w, v = np.linalg.eig(cvmat)
        index = np.argmax(w)
        return [-np.arctan2(v[index][1], v[index][0]),
                -np.arctan2(v[1-index][1], v[1-index][0])]

    def spiralize_batch(self, x, alpha=None):
        '''
        Transforms a given batch into a spiral-shaped distribution by applying  the spiral transformation 
        x' = x cos(alpha r) - y sin(alpha r)
        y' = x sin(alpha r) + y cos(alpha r)
        '''
        if alpha == None:
            alpha = self.alpha
        r = np.sqrt(np.sum(x**2, -1))

        x_sp = x[:, 0]*np.cos(alpha*r) - x[:, 1]*np.sin(alpha*r)
        y_sp = x[:, 0]*np.sin(alpha*r) + x[:, 1]*np.cos(alpha*r)

        return np.array([x_sp, y_sp]).T

    def sample(self, N_batch):
        ''' Samples from the spiral distribution
        '''
        return self.spiralize_batch(np.random.multivariate_normal([0, 0], self.cov, N_batch))

    def P_x(self, x):
        ''' Returns the theoretical P_x(x) of the given array of points with shape [N_samples, 2]
        '''
        x_sp = self.spiralize_batch(x, -self.alpha)
        return 1/np.sqrt((2*np.pi)**2*np.linalg.det(self.cov))*np.exp(-0.5*np.diag(np.dot(np.dot(x_sp, self.inv_cov), x_sp.T)))

    def plot_feature(self, feature_funct, label="Feature", N_levels=15):
        '''
        Plots the spiral distribution and the contour lines of a given feature_funct. 
        Please provide the function(samples) and not the data itself.
        '''
        x1min = -7
        x1max = 7
        x2min = -7
        x2max = 7

        N = 150
        mypoints1 = np.linspace(x1min, x1max, N)
        mypoints2 = np.linspace(x2min, x2max, N)
        mymeshx, mymeshy = np.meshgrid(mypoints1, mypoints2)
        myin = np.array([mymeshx.flatten(), mymeshy.flatten()]).T
        myout = self.P_x(myin).reshape([N, N])

        Samples = self.sample(10000)
        mask = (Samples[:, 0] < x1max-0.3)*(Samples[:, 0] > x1min+0.3) * \
            (Samples[:, 1] < x2max-0.3)*(Samples[:, 1] > x2min+0.3)
        Feature = feature_funct(myin)[0]
        Feature = Feature.reshape([N, N])

        plt.figure(figsize=[10, 10])
        plt.title(label)
        plt.gca().set_aspect('equal')
        plt.contourf(myout, 1000, cmap=plt.cm.BrBG_r,
                     extent=(x1min, x1max, x2min, x2max))
        plt.scatter(Samples[mask, 0], Samples[mask, 1],
                    alpha=0.1, s=3, color="orange")
        plt.contour(mymeshx, mymeshy, Feature, N_levels, colors="white")
        plt.xlabel(r"$x_1$")
        plt.xlabel(r"$x_2$")
        plt.show()


class WavePacket:
    '''
    One dimensional field with some background noise and a fixed-shaped wave packet (Gaussian)
    '''

    def produce_Wave_Packet(self,
                            n_pixels=100,
                            n_samples=10000,
                            width=3.0,
                            noise=0.0,
                            noisy_amplitude=None,
                            pos_range=None,
                            colored_noise=None,
                            pos_avg_span=False):
        '''
        One dimensional field with length n_pixels:
        - width = shape of the packet e^(-((j-jbar)/width)^2)
        - noise = std of the background noise
        - noisy_amplitude = dA: the wave packet has amplitude (1+dA), if not given dA=0
        - pos_range = possible position of the center of the wave packet (uniformly distributed in this interval). If n_pixels=100, we suggest [30,70]. This is to avoid possible boundary effects (the wavepacket is cut)
        - colored_noise
        - pos_avg_span = True: instead of placing the wavepacket uniformly, the position is sequential
        '''

        x = np.array(range(n_pixels))

        #####################
        # Background noise
        if colored_noise == None:
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
        if pos_range == None:
            pos_range = [0, n_pixels]
        pos_width = pos_range[1] - pos_range[0]

        if pos_avg_span == False:
            pos = pos_range[0] + np.random.random(n_samples)*pos_width
        else:
            pos = pos_range[0] + np.linspace(0., 1., n_samples)*pos_width

        # Amplitude fluctuation
        if noisy_amplitude == None:
            amp = 1.0
        else:
            # shape [N_samples, N_pixels]
            amp = 1 + noisy_amplitude*np.random.randn(n_samples)[:, np.newaxis]

        wave_packet = amp * \
            np.exp(- (x[np.newaxis, :] - pos[:, np.newaxis])**2/width**2)

        #####################
        # Final Samples
        Samples = background_noise + wave_packet

        self.last_pos = pos
        return Samples

    def plot(self, samples, N_samples=100):
        '''
        Plots the first N_samples. Color represents the intensity of the field, X is the coordinate of the field (i.e. j) and Y represents the number of samples.
        '''
        plt.figure(figsize=[10, 10])
        plt.imshow(samples[0:N_samples, :])
        plt.xlabel("x")
        plt.ylabel("# sample")
        plt.show()


class Simulator:
    '''
     This class simulates a system of particles in a potential (quartic)
     Initialize the class with required parameters and then perform the simulation to extract samples with the function simulate()
    '''

    def __init__(self, n_particles, n_batch, force_params, spring_K, gamma, temperature, x0, dx0, timestep, max_time):
        '''
        Required arguments:
        - n_particles
        - n_batch
        - force_params = [f1, f2, f3] -> used force F = f1 x + f2 x^2 + f3 x^3
        - spring_K
        - gamma:  damping
        - temperature: Noise_Force = np.sqrt(2*gamma*temperature/timestep)
        - x0 : initial center of the cloud of particles
        - dx0: initial spread
        - timestep
        - max_time
        '''
        self.N_particles = n_particles
        self.N_batch = n_batch
        self.N_total = n_particles*n_batch
        self.Force1 = force_params[0]
        self.Force2 = force_params[1]
        self.Force3 = force_params[2]
        self.Gamma = gamma
        self.Noise_Force = np.sqrt(2*gamma*temperature/timestep)
        self.Spring_K = spring_K

        self.Temperature = temperature
        self.x0 = x0
        self.dx0 = dx0
        self.timestep = timestep
        self.Max_Time = max_time

        self.n_timesteps = int(max_time/timestep)

    def Force(self, x):
        return self.Force1*x + self.Force2*(x**2) + self.Force3*(x**3)

    def rhs(self, X):

        x = X[0:self.N_total]
        v = X[self.N_total:2*self.N_total]

        dX = np.zeros(2*self.N_total)
        dX[0:self.N_total] = v
        # the individual forces on the particles (the same for all)
        dX[self.N_total:2*self.N_total] = self.Force(x)-self.Gamma*v

        # the following will give [x_CM1,x_CM1,x_CM1,x_CM2,x_CM2,x_CM2]
        # if we have two groups of three particles (and x_CM is their
        # center-of-mass position, multiplied by N_particles, in other
        # words, the sum over all particle coordinates)
        x_CM = np.sum(x.reshape([self.N_batch, self.N_particles]), axis=1).repeat(
            self.N_particles)
        # the force produced by the springs between the particles:
        dX[self.N_total:2*self.N_total] += self.Spring_K * \
            (x_CM-self.N_particles*x)

        # the noise force
        dX[self.N_total:2*self.N_total] += self.Noise_Force * \
            np.random.randn(self.N_total)

        return(dX)

    def simulate(self):
        """
        Simulate n_batch clouds of n_particles, interacting via
        springs (constant spring_K), and subject to a force field
        f1*x+f2*(x**2)+f3*(x**3), that derives from a quartic potential.
        (where force_params=[f1,f2,f3])

        There is damping gamma, and a noise force that matches the
        'temperature'. Initially, the particles are placed around
        position x0, with spread dx0.

        We use a simple Euler solver, with a 'timestep', running
        up to max_time.

        Returns an array Xs of size [n_timesteps,2*N_total],
        where N_total=n_batch*n_particles, and where
        for a given timestep j, we have that Xs[j,0:N_total] are
        the coordinates, while Xs[j,N_total:2*N_total] are the
        velocities.

        After about a damping time 1/Gamma, the system should
        reach thermal equilibrium at the given 'temperature'.

        Have fun!
        """
        # random initialization
        # all particles around x0, with some spread
        X = np.zeros(self.N_total*2)
        X[0:self.N_total] = self.x0+self.dx0*np.random.randn(self.N_total)
        X[self.N_total:self.N_total *
            2] = np.sqrt(self.Temperature)*np.random.randn(self.N_total)

        # Euler solver:

        Xs = np.zeros([self.n_timesteps, 2*self.N_total])
        for j in range(self.n_timesteps):
            X += self.timestep*self.rhs(X)
            Xs[j, :] = X  # store results

        ts = np.array(range(self.n_timesteps))*self.timestep
        return(Xs, ts)

    def extract_batches(self, Xs):
        """
        Given array Xs of shape [n_timesteps,2*N_particles*N_batch],
        return array of shape [n_timesteps*N_batch,N_particles] that
        contains all the samples in configuration space,
        of all batches at all times
        """
        return(Xs[:, 0:self.N_total].reshape([np.shape(Xs)[0]*self.N_batch, self.N_particles]))

    def plot(self, ts, Xs, N=None):
        plt.figure()
        plt.xlabel("t")
        plt.ylabel("x")

        if N == None:
            N = self.N_particles

        for n in range(N):
            plt.plot(ts, Xs[:, n], color=[
                     1, n*1.0/self.N_particles, 0.5], alpha=0.5, linewidth=1)
        plt.show()

    def plot_poshist(self, Xs):
        plt.figure()
        plt.hist(Xs[:, 0:self.N_total].flatten(), bins=100, density=True)
        plt.title("X distribution")
        plt.xlabel("x")
        plt.ylabel(r"$P_x$")
        plt.show()


class RandomMolecules:
    ''' This class just simulates a 1d gas of molecules
    '''

    def __init__(self, N_particles_molecule, N_particles_gas, molecule_size, molecule_spread, gas_spread):
        self.N_particles_molecule = N_particles_molecule
        self.N_particles_gas = N_particles_gas
        self.molecule_size = molecule_size
        self.molecule_spread = molecule_spread
        self.gas_spread = gas_spread

        self.N_particles = N_particles_molecule + N_particles_gas

    def sample(self, n_samples):
        Samples = np.zeros([n_samples, self.N_particles])

        # the molecules
        Samples[:, 0:self.N_particles_molecule] =\
            self.molecule_spread*(np.random.random([n_samples])[:, np.newaxis]-0.5) +\
            self.molecule_size * \
            np.random.randn(n_samples, self.N_particles_molecule)

        # the gas particles
        Samples[:, self.N_particles_molecule:] =\
            self.gas_spread * \
            (np.random.random([n_samples, self.N_particles_gas])-0.5)
        return Samples

    def plot(self, Samples, N=10):
        plt.figure(figsize=[10, 8], dpi=90)
        plt.title("Random Molecules in a Gas")
        for n in range(N):
            plt.scatter(Samples[n, :], np.full(
                self.N_particles, n), s=3, alpha=0.2)
        plt.xlabel("x")
        plt.ylabel("# sample")
        plt.show()
