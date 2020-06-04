'''
lennardjones.py

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
Renormalized Mutual Information - Example of a "solid drop"

This package generates samples of a T=0 drop.
Particles interact in a Lennard-Jones type of potential (n=1 by default) and are confined in a drop with elliptical shape.
The ellipse usually has DeltaR and theta (deformation and orientation) randomly chosen from a gaussian and uniform distribution.

'''


import numpy as np
import matplotlib.pyplot as plt


def plot_drop(i, particles, drop, interactions, ax=None, color="blue", delta_r=None, radius_0=None, theta=None, show3d=False):
    '''
    Plot a single sample
    - i = index of the sample in the 'particles' array (shape=[N_samples, N_particles, 2])
    - particles
    - drop, interactions: settings to generate the samples
    - show3d = enable 3d mode (paper publication quality)
    '''
    size = 5
    dpi = 100
    if ax is None:
        plt.figure(figsize=(size, size), dpi=dpi)
        ax = plt.gca()
        do_plt_show = True
    else:
        do_plt_show = False

    r_x = particles[i, :, 0]
    r_y = particles[i, :, 1]
    scale_scatter = (interactions.d_coll/2)
    if show3d:
        nsteps = 10
        rad0_pix = 2*5.0
        rad0 = 2*0.02
        for n in range(nsteps):
            factor = (1.0*(n+1))/nsteps
            rad = rad0_pix*(1-0.9*factor)
            shift = (rad0/np.sqrt(2))*factor

            Red = np.tanh(factor*3)
            Green = factor**2
            Blue = factor**3

            ax.scatter(r_x-shift, r_y+shift, s=(scale_scatter*rad) **
                       2*size*dpi, color=np.array([Red, Green, Blue]))
    else:
        ax.scatter(r_x, r_y, s=(scale_scatter**2)*size*dpi, color=color)

    if drop.deltaR[i] is not None:
        phi = np.linspace(0, 2*np.pi, 100)
        xp = (drop.R+drop.deltaR[i, 0, 0])*np.cos(phi)
        yp = (drop.R**2)*np.sin(phi)/(drop.R+drop.deltaR[i, 0, 0])
        ctheta = np.cos(drop.theta[i, 0, 0])
        stheta = np.sin(drop.theta[i, 0, 0])

        x = xp*ctheta-yp*stheta
        y = xp*stheta+yp*ctheta
        plt.plot(x, y, '--', color="black")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    if do_plt_show:
        plt.show()


def norm(point):
    return np.sqrt(norm2(point))


def norm2(point):
    return np.sum(point**2, -1)


class Drop:
    '''
        Describes the shape of the solid drop.
        - R is the radius
        - deltaR is the deformation. Since we describe an uncompressible drop, the axis of the ellipse are (R+DR) and R^2/(R+DR)
        - theta: rotation of the ellipose
        - ampl_confinement: strength of the confinement potential (the force that brings the particles back inside the drop)
    '''

    def __init__(self, R, deltaR, theta, ampl_confinement):
        self.R = R
        self.deltaR = deltaR

        self.axisA = (self.R + self.deltaR).flatten()
        self.axisB = (self.R**2/(self.R + self.deltaR)).flatten()

        self.theta = theta
        self.ctheta = np.cos(theta)[:, :, 0]
        self.stheta = np.sin(theta)[:, :, 0]

        self.A_conf = ampl_confinement

    def rotatePoint(self, point):
        # Point with shape [N_particles, N_particles 2]
        pnew = np.zeros([len(point), np.shape(point)[1], 2])
        pnew[:, :, 0] = point[:, :, 0]*self.ctheta + point[:, :, 1]*self.stheta
        pnew[:, :, 1] = -point[:, :, 0] * \
            self.stheta + point[:, :, 1]*self.ctheta
        return pnew
    def isInside(self, point):
        # we rotate back the point to the frame parallel to the axis of the ellipse
        rotatedPoint = self.rotatePoint(point)
        return ((rotatedPoint[:, :, 0]/self.axisA[:, None])**2 + (rotatedPoint[:, :, 1]/self.axisB[:, None])**2 < 1)

    def isOutside(self, point): return 1-self.isInside(point)

    # Calculate the confinement potential
    # Return the contribution to the potential of the single particle
    def V_vect(self, points):
        return self.A_conf*norm(points)*self.isOutside(points)

    # Return the complete potential
    def V(self, points): return np.sum(self.V_vect(points), -1)

    # Calculate the confinement force (bring back inside)
    def F(self, points):
        return -self.A_conf/norm(points)[:, :, None]*points*self.isOutside(points)[:, :, None]


class Interactions:
    '''
        Describes the interactions between the particles. In this example, we implement a Lennard-Jones type of potential.
        In particular, n=1 in this case.
        - d_collision is the small-length cutoff of the potential (to avoid numerical divergence)
        - d_attraction is the equilibrium distance between two particles
    '''

    def __init__(self, d_collision, d_attraction):
        self.d_coll = d_collision
        self.d_attr = d_attraction

    # it returns a matrix with shape [N_samples, N_particles, N_particles, 2]
    def get_distance_matrix(self, points):
        return points[:, :, np.newaxis, :]-points[:, np.newaxis, :, :]

    def isColliding(self, distances_norm):
        isColl = distances_norm < self.d_coll
        for i in range(len(isColl)):
            np.fill_diagonal(isColl[i], 0)
        return isColl

    def V_vect(self, distances):
        distances_norm2 = norm2(distances)
        distances_norm = np.sqrt(distances_norm2)
        isColliding = self.isColliding(distances_norm)
        # Collision term proportional to d**2 (cutoff)
        v_colliding = -distances_norm2/self.d_coll**2 + 1.5+0.5 * \
            (self.d_attr/self.d_coll)**(2*n) - (self.d_attr/self.d_coll)**n
        v_colliding *= isColliding
        # Interaction potential: d - ln d
        v_interact = 0.5*self.d_attr**(2*n)/(np.identity(np.shape(distances_norm2)[1])[None, :, :]+distances_norm2)**n - self.d_attr**n/(
            np.identity(np.shape(distances_norm2)[1])[None, :, :]+distances_norm2)**(n/2) + 0.5
        v_interact *= (1 - isColliding)

        v = v_colliding + v_interact
#         diag = np.einsum('ijj->ij', v)
#         diag[:,:] = 0
        for i in range(len(v)):
            np.fill_diagonal(v[i], 0)
        return v

    def get_vshape(self, dmax, plot=True):
        '''
            function to plot the shape of the interaction potential (as a function of the distance between two particles)
            - dmax is the maximum distance to consider (starting from 0)
        '''
        dists_label = np.linspace(0.001, dmax, 200)
        dists = np.concatenate([[0], np.linspace(0.001, dmax, 200)])/np.sqrt(2)
        dists_xy = np.stack([dists, dists], 1)
        dist_mat = self.get_distance_matrix(dists_xy[None, :, :])
        v_vect = self.V_vect(dist_mat)[0, 0, 1:]
        if plot:
            plt.figure()
            plt.plot(dists_label, v_vect)
            plt.ylim([0, 2])
            plt.show()
        return dists_label, v_vect

    def V(self, distances):
        return np.sum(self.V_vect(distances), (1, 2))

    def F_mat(self, distances):
        distances_norm2 = norm2(distances)
        distances_norm = np.sqrt(distances_norm2)
        isColliding = self.isColliding(distances_norm)[:, :, :, None]
        # Repulsion force when a collision happens
        f_colliding = (2/self.d_coll**2)*isColliding
        # Interaction force
        ident = np.identity(np.shape(distances)[1])[None, :, :]
        d = (ident+distances_norm)
        dn = (d**(n+2))[:, :, :, None]
        d2 = (ident+distances_norm2)
        d2n = (d2**(n+1))[:, :, :, None]

        f_interact = n*(self.d_attr**(2*n)/d2n -
                        self.d_attr**n/dn)*(1-isColliding)
        f = (f_colliding + f_interact)*distances

        diag = np.einsum('ijj->ij', f[:, :, :, 0])
        diag[:, :] = 0

        diag2 = np.einsum('ijj->ij', f[:, :, :, 1])
        diag2[:, :] = 0

        return f

    def F(self, distances):
        return np.sum(self.F_mat(distances), 1)


###
# Beginning of sampling code
###
n = 1  # shape of Lennard Jones-like potential
eta = 10e-7


def sample(N_samples, N_particles, drop):
    '''
        Generate ellipses with random deformations and orientations
        Randomly place N_particles inside (with uniform distribution)
    '''
    max_dist = (np.maximum(drop.axisA, drop.axisB))[:, None, None]
    particles = (np.random.random(
        size=(N_samples, 90*N_particles, 2))-0.5)*2*1.5*max_dist
    w = drop.isInside(particles)

    newpart = np.zeros([N_samples, N_particles, 2])
    for i in range(N_samples):
        newpart[i] = particles[i, w[i]][:N_particles]
    return newpart


def particles_to_samples(particles):
    '''
        Function used just to reshape the particles array from [N_samples,N_particles,2] to [N_samples,2*N_particles] (used in renormalized mutual information module)
    '''
    N_samples = np.shape(particles)[0]
    N_particles = np.shape(particles)[1]

    samples = np.zeros([N_samples, 2*N_particles])
    samples[:, :N_particles] = particles[..., 0]
    samples[:, N_particles:] = particles[..., 1]
    return samples


def relax(particles, drop, interactions, N_steps=1000, plot=False):
    '''
        Perform relaxation to equilibrium (starting from some other distribution)
        This is just gradient descent through the potential.
        We can add a stochastic term to simulate a finite temperature; otherwise T=0.
    '''
    N_samples = np.shape(particles)[0]
    N_particles = np.shape(particles)[1]

    for i in range(N_steps):
        distances = interactions.get_distance_matrix(particles)
#         potentials.append(interactions.V(distances)[0])

        particles -= eta*interactions.F(distances)
        particles += eta*drop.F(particles)
        particles += 10**(-5)*np.random.randn(N_samples,N_particles,2)
        if i % 100 == 0:
            if plot:
                plot_drop(0, particles, drop, interactions, show3d=True)
    return particles
