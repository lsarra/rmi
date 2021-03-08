
# lennardjones.py

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
import pickle
"""This package generates samples of a drop.
Particles interact in a Lennard-Jones type of potential (n=6 by default) and are confined in a drop with elliptical shape.
The ellipse usually has DeltaR and theta (deformation and orientation) randomly chosen from a uniform distribution.
"""

def load_dataset(label, max_deltaR=None):
    """Load a liquid drop dataset

    Files should be located in the `dataset` folder.
    Required files are PARTICLES, DROP and INTERACTIONS.

    Args:
        label (str): 
        max_deltaR (float, optional): Maximum drop deformation. Used to load only a part of the dataset. Defaults to None.

    Returns:
        samples (array_like): [N_samples, 2xN_particles] particle positions in each sample
        water (lj.Drop): liquid drop parameters
        interactions (lj.Interactions): interaction parameters
    """

    # Files should be located in the datasets directory
    particles = np.load("datasets/PARTICLES_"+label+".npy", allow_pickle=True)

    with open("datasets/DROP_"+label, 'rb') as pickle_file:
        water = pickle.load(pickle_file)

    with open("datasets/INTERACTIONS_"+label, 'rb') as pickle_file:
        interactions = pickle.load(pickle_file)

    samples = particles_to_samples(particles)

    if max_deltaR is not None:
        whr = water.deltaR.flatten() < max_deltaR
        samples = samples[whr]
        water.deltaR = water.deltaR[whr]
        water.theta = water.theta[whr]

    return samples, water, interactions


def get_packing_distance(ellipse_R, N_particles):
    """Calculates the optimal equilibrium distance 
    so that the particles occupate the entire ellipse

    Args:
        ellipse_R (float): radius of the ellipse
        N_particles (int): number of particles to place inside of the ellipse

    Returns:
        d_eq: equilibrium distance
    """
    return np.sqrt(2*np.pi/(np.sqrt(3)*N_particles))*ellipse_R


def separate_particles(samples, index=None):
    """Get particle coordinates from array (just reshape the array)

    Args:
        samples (array_like): [N_samples, 2xN_particles]
        index (int, optional): Where to split, (usually at the half of the array). Defaults to None.

    Returns:
        px (array_like): [N_samples, N_particles] first coordinate of the particles
        py (array_like): [N_samples, N_particles] second coordinate of the particles
    """

    if index is None:
        index = int(len(samples.shape[1])/2)
    px = samples[:, :index]
    py = samples[:, index:]
    return px, py


def particles_to_samples(particles):
    """Reshape particle dataset

    Function used just to reshape the particles array from [N_samples,N_particles,2] to [N_samples,2*N_particles]

    Args:
        particles (array_like): [N_samples, N_particles, 2]

    Returns:
        samples: [N_samples, 2xN_particles]
    """

    N_samples = np.shape(particles)[0]
    N_particles = np.shape(particles)[1]

    samples = np.zeros([N_samples, 2*N_particles])
    samples[:, :N_particles] = particles[..., 0]
    samples[:, N_particles:] = particles[..., 1]
    return samples

############################################
# Some sample features


def variance(samples):
    """Handcrafted feature: Variance

    Feature (sum x_1^2, sum x_2^2)

    Args:
        samples (array_like): [N_samples, 2xN_particles]

    Returns:
        feature (array_like): [N_samples, 2]
        grad_feature (array_like): [N_samples, 2, 2xN_particles]
    """
    N_samples = np.shape(samples)[0]
    N_particles = int(np.shape(samples)[1]/2)
    px, py = separate_particles(samples, N_particles)

    varx = np.mean(px**2, -1)
    vary = np.mean(py**2, -1)

    grads = np.zeros([N_samples, 2, 2*N_particles])

    grad_varx = 2*px/N_particles
    grad_vary = 2*py/N_particles

    var = np.array([varx, vary]).T
    grads[:, 0, :N_particles] = grad_varx
    grads[:, 1, N_particles:] = grad_vary

    return var, grads


def handcrafted_feature(samples):
    """Handcrafted feature

    Feature (sum x_1^2, sum x_1 x_2)

    Args:
        samples (array_like): [N_samples, 2xN_particles]

    Returns:
        feature (array_like): [N_samples, 2]
        grad_feature (array_like): [N_samples, 2, 2xN_particles]
    """
    N_samples = np.shape(samples)[0]
    N_particles = int(np.shape(samples)[1]/2)
    px, py = separate_particles(samples, N_particles)

    corr = np.mean(px*py, -1)
    varx = np.mean(px**2, -1)

    feat = np.array([varx, corr]).T

    grads = np.zeros([N_samples, 2, 2*N_particles])

    grad_varx = 2*px/N_particles

    grads[:, 0, :N_particles] = grad_varx
    grads[:, 0, N_particles:] = 0

    grad_corrx = py/N_particles
    grad_corry = px/N_particles

    grads[:, 1, :N_particles] = grad_corrx
    grads[:, 1, N_particles:] = grad_corry

    return feat, grads


def norm(point):
    """Norm of a point

    Args:
        point (array_like): [..., 2] last dimension of the array should be the coordinate

    Returns:
        (array_like): [...] squared norm on the last dimension
    """
    return np.sqrt(norm2(point))


def norm2(point):
    """Squared norm of a point

    Args:
        point (array_like): [..., 2] last dimension of the array should be the coordinate

    Returns:
        (array_like): [...] squared norm on the last dimension
    """
    return np.sum(point**2, -1)


class Drop:
    """Describes the shape of the solid drop.

    Drops have elliptical shape, whose semiaxes are given by (R+DR) and R^2/(R+DR).
    In this way, the drop is uncompressible (i.e. all drops have same area).
    The orientation of the drop is the rotation angle of the ellipse.

    """

    def __init__(self, R, deltaR, theta, ampl_confinement):
        """Initialize the class

        Args:
            R (float): radius of the drop (if it was a circle)
            deltaR (array_like): [N_samples, 1, 1] deformation of the drop. 
            theta (array_like): [N_samples, 1, 1] orientation of the drop
            ampl_confinement (float): strength of the drop wall potential
        """
        self.R = R
        self.deltaR = deltaR

        self.axisA = (self.R + self.deltaR).flatten()
        self.axisB = (self.R**2/(self.R + self.deltaR)).flatten()

        self.theta = theta
        self.ctheta = np.cos(theta)[:, :, 0]
        self.stheta = np.sin(theta)[:, :, 0]

        self.A_conf = ampl_confinement

    def rotatePoint(self, point, dir=+1):
        """Rotate a point in the direction of the drop

        Args:
            point (array_like): [N_samples, N_particles, 2]
            dir (int, optional): Rotation direction (clockwise or anti-clockwise). Defaults to +1.

        Returns:
            (array_like): [N_samples, N_particles, 2] rotated point 
        """
        pnew = np.zeros([len(point), point.shape[1], 2])
        pnew[:, :, 0] = point[:, :, 0]*self.ctheta + point[:, :, 1]*self.stheta*dir
        pnew[:, :, 1] = -point[:, :, 0] * \
            self.stheta*dir + point[:, :, 1]*self.ctheta
        return pnew

    def isInside(self, point):
        """Check whether inside the drop or not

        Args:
            point (array_like): [N_samples, N_particles, 2]

        Returns:
            (array_like): [N_samples, N_particles] (bool) 
        """
        # we rotate back the point to the frame parallel to the axis of the ellipse
        rotatedPoint = self.rotatePoint(point)
        # we check if each point is inside the associated liquid drop
        return ((rotatedPoint[:, :, 0]/self.axisA[:, None])**2 + (rotatedPoint[:, :, 1]/self.axisB[:, None])**2 < 1)

    def isOutside(self, point):
        """Check if a point is outside the drop

        Just `1 - isInside(point)`

        Args:
            point (array_like): [N_samples, N_particles, 2]

        Returns:
            (array_like): [N_samples, N_particles] (bool) 
        """
        return 1-self.isInside(point)

    def V_vect(self, points):
        """Confinement potential (per particle)

        The wall potential is `A_conf * distance_from_origin`
        and it applies only if a particle is outside.

        Args:
            points (array_like): [N_samples, N_particles, 2]

        Returns:
            (array_like): [N_samples, N_particles] contribution to the potential of the single particle
        """
        return self.A_conf*norm(points)*self.isOutside(points)

    def V(self, points):
        """Confinement potential

        Args:
            points (array_like): [N_samples, N_particles, 2]

        Returns:
            (array_like): [N_samples] drop potential of the entire sample 
        """
        return np.sum(self.V_vect(points), -1)

    def F(self, points):
        """Calculate the confinement force
        (i.e. the force that brings a particle back inside the drop)
        It should act only if the particle is outside the drop.

        Args:
            points (array_like): [N_samples, N_particles, 2]

        Returns:
            (array_like): [N_samples, N_particles, 2]
        """

        # Perform calculation in the frame parallel to the ellipse
        # and then rotate it back
        rotatedP = self.rotatePoint(points)

        surface_pot = np.sqrt((rotatedP[..., 0]/self.axisA[:, None])**2 + (rotatedP[..., 1]/self.axisB[:, None])**2)

        # Force = -grad potential
        # swapaxes is necesary to make the created
        # force array of shape [N_samples, N-particles, 2]

        surface_force = - 1/surface_pot[..., None]*np.array([
             rotatedP[..., 0]/self.axisA[:, None]**2, 
             rotatedP[..., 1]/self.axisB[:, None]**2
            ]).swapaxes(0, 2).swapaxes(0, 1) 

        # Rotate back to the original frame
        surface_force = self.rotatePoint(surface_force, -1)

        # Force is scaled with A_conf and applied only if the particle is outside
        return self.A_conf*surface_force*self.isOutside(points)[:, :, None]


class Interactions:
    """        Describes the interactions between the particles. 
    
    In this example, we implement a Lennard-Jones type of potential.
    In particular, n=6 in this case (defined below as global variable).
    """

    def __init__(self, d_collision, d_attraction, n=6):
        """Initialize the class

        Args:
            d_collision (float):  small-length cutoff of the potential (to avoid numerical divergence)
            d_attraction (float): equilibrium distance between two particles
            n (int): power of the Lennard Jones potential. Defaults to 6.
        """
        self.d_coll = d_collision
        self.d_attr = d_attraction
        self.n = n

    # it returns a matrix with shape 
    def get_distance_matrix(self, points):
        """All the distances between particles (coordinate difference)

        Args:
            points (array_like): [N_samples, N_particles, 2]

        Returns:
            (array_like): [N_samples, N_particles, N_particles, 2]
        """
        return points[:, :, np.newaxis, :]-points[:, np.newaxis, :, :]

    def isColliding(self, distances_norm):
        """Check which particles are collading

        Colliding particles are those whose distance
        is smaller than d_coll

        Args:
            distances_norm (array_like): [N_samples, N_particles, N_particles] distances between particles 
                                        (each particle with all the others)

        Returns:
            (array_like): [N_samples, N_particles, N_particles] bool, whether they collide or not
        """
        isColl = distances_norm < self.d_coll

        # A particle does not collide with itself
        for i in range(len(isColl)):
            np.fill_diagonal(isColl[i], 0)
        return isColl

    def V_vect(self, distances):
        """Particle-Particle interaction (per particle)

        Per particle Lennard-Jones potential

        Args:
            distances (array_like): [N_samples, N_particles, N_particles, 2]

        Returns:
            (array_like): [N_samples, N_particles, N_particles]
        """
        distances_norm2 = norm2(distances)
        distances_norm = np.sqrt(distances_norm2)
        isColliding = self.isColliding(distances_norm)

        # Collision term proportional to d**2 (cutoff)
        v_colliding = -distances_norm2/self.d_coll**2 + 1.5+0.5 * \
            (self.d_attr/self.d_coll)**(2*self.n) - (self.d_attr/self.d_coll)**self.n
        v_colliding *= isColliding

        # Interaction potential: d - ln d
        v_interact = 0.5*self.d_attr**(2*self.n)/(np.identity(np.shape(distances_norm2)[1])[None, :, :]+distances_norm2)**self.n - self.d_attr**self.n/(
            np.identity(np.shape(distances_norm2)[1])[None, :, :]+distances_norm2)**(self.n/2) + 0.5
        v_interact *= (1 - isColliding)

        v = v_colliding + v_interact

        # A particle does not interact with itself
        for i in range(len(v)):
            np.fill_diagonal(v[i], 0)
        return v

 
    def V(self, distances):
        """Interaction potential

        Args:
            distances (array_like): [N_samples, N_particles, N_particles, 2]

        Returns:
            (array_like): [N_samples] total interaction potential for each drop
        """
        return np.sum(self.V_vect(distances), (1, 2))

    def F_mat(self, distances):
        """Calculate force due to each other particle

        Args:
            distances (array_like): [N_samples, N_particles, N_particles, 2]

        Returns:
            force matrix (array_like): [N_samples, N_particles, N_particles, 2]
        """
        distances_norm2 = norm2(distances)
        distances_norm = np.sqrt(distances_norm2)
        isColliding = self.isColliding(distances_norm)[:, :, :, None]

        # Repulsion force when a collision happens
        f_colliding = (2/self.d_coll**2)*isColliding
        
        # Interaction force
        ident = np.identity(np.shape(distances)[1])[None, :, :]
        d = (ident+distances_norm)
        dn = (d**self.n)[:, :, :, None]
        d2 = (ident+distances_norm2)
        d2n = (d2**(self.n+1))[:, :, :, None]

        f_interact = self.n*self.d_attr**self.n*(self.d_attr**self.n-dn)/(d2n + 10e-50)*(1-isColliding)

        # Total Force
        f = (f_colliding + f_interact)*distances

        # Remove self-interaction
        diag = np.einsum('ijj->ij', f[:, :, :, 0])
        diag[:, :] = 0

        diag2 = np.einsum('ijj->ij', f[:, :, :, 1])
        diag2[:, :] = 0

        return f

    def F(self, distances):
        """Total force on each particle

        Args:
            distances (array_like): [N_samples, N_particles, N_particles, 2]

        Returns:
            force (array_like): [N_samples, N_particles, 2]
        """
        return np.sum(self.F_mat(distances), 1)

    def get_vshape(self, dmax, plot=True):
        """ Shape of Lennard-Jones potential
        
        Function to plot the shape of the interaction potential 
        (as a function of the distance between two particles)

        Args:
            dmax (float):  maximum distance to consider (starting from 0)
            plot (bool, optional): Show the plot (or only evaluate the potential). Defaults to True.

        Returns:
            dists (array_like): [200] distances for which potential is evaluated
            v (array_like): [200] value of the potential at each distance
        """
        '''

        '''
        dists_label = np.linspace(0.001, dmax, 200)
        dists = np.concatenate([[0], np.linspace(0.001, dmax, 200)])/np.sqrt(2)
        dists_xy = np.stack([dists, dists], 1)
        dist_mat = self.get_distance_matrix(dists_xy[None, :, :])
        v_vect = self.V_vect(dist_mat)[0, 0, 1:]

        if plot:
            _, ax = plt.subplots(dpi=150)
            plt.title("Lennard-Jones Potential")
            plt.xlabel("interaction distance")
            plt.ylabel("potential")
            plt.plot(dists_label, v_vect, c="blue")
            plt.axvline(x=self.d_coll, color="red", linestyle="dashed", linewidth=1)
            plt.annotate(r"$d_{coll}$", (self.d_coll, np.min(v_vect)), textcoords="offset points", xytext=(15, -5), ha='center', color="red")
            plt.axvline(x=self.d_attr, color="black", linestyle="dashed", linewidth=1)
            plt.annotate(r"$d_{eq}$", (self.d_attr, np.min(v_vect)), textcoords="offset points", xytext=(15, -5), ha='center')
            plt.semilogy()

            axins = ax.inset_axes([0.67, 0.67, 0.3, 0.3])
            axins.set_xlim(0.15, 0.4)
            axins.set_ylim(-0.2, 2)

            axins.plot(dists_label, v_vect, c="blue")
            axins.axvline(x=self.d_attr, color="black", linestyle="dashed", linewidth=1)
            axins.annotate(r"$d_{eq}$", (self.d_attr, np.min(v_vect)), textcoords="offset points", xytext=(15, -5), ha='center')

            plt.show()

        return dists_label, v_vect



#########################################
# Beginning of sampling code
###

def sample(N_samples, N_particles, drop):
    """Generate random ellipses

    Generate ellipses with random deformations and orientations
    Randomly place N_particles inside (with uniform distribution)

    Particles in each sample are uncorrelated. 
    To thermalize them according to a Lennard-Jones interaction,
    one can perform relaxation using `relax` function.

    Args:
        N_samples (int): number of samples
        N_particles (int): number of particles for each sample
        drop (lj.Drop): properties of each liquid drop

    Returns:
        (array_like): [N_samples, N_particles, 2] generated samples
    """

    max_dist = (np.maximum(drop.axisA, drop.axisB))[:, None, None]
    particles = (np.random.random(
        size=(N_samples, 90*N_particles, 2))-0.5)*3*max_dist
    w = drop.isInside(particles)

    newpart = np.zeros([N_samples, N_particles, 2])
    for i in range(N_samples):
        newpart[i] = particles[i, w[i]][:N_particles]
    return newpart


def relax(particles, drop, interactions,
          N_steps=1000, eta=10e-7, plot=False,
          interaction=True, border=True, temperature=True):
    """Relaxation of a drop

    Thermalize each configuration with a Lennard-Jones interaction
    between each particle. Particles will end up uniformly occupying
    the ellipse, without overlapping each other.

    This is just gradient descent through the potential.
    We can add a stochastic term to simulate a finite temperature; otherwise T=0.

    PLEASE NOTE: thermalization can be quite slow, and take a high
    amount of memory (RAM) if the number of samples is large.
    For testing purposes one could avoid thermalization of the samples.

    Args:
        particles (array_like): [N_samples, N_particles, 2]
        drop (lj.Drop): drop parameters
        interactions (lj.Interactions): interaction parameters
        N_steps (int, optional): Thermalization steps. Defaults to 1000.
        eta (float, optional): amplitude of each step. Defaults to 10e-7.
        plot (bool, optional): Whether to show the evolution of the samples during thermalization. Defaults to False.
        interaction (bool, optional): Apply particle interaction. Defaults to True.
        border (bool, optional): Apply drop confinement. Defaults to True.
        temperature (bool, optional): Keep some finite temperature 
            (particles keep moving and don't perfectly align on the minimum of the potential). Defaults to True.

    Raises:
        Exception: due to numerical inaccuracies or if some particles somehow escapes the drop (it should not happen!)

    Returns:
        (array_like): thermalized particles
    """
    # global potentials # just for debug
    # potentials = []
    
    N_samples = np.shape(particles)[0]
    N_particles = np.shape(particles)[1]
    for i in range(N_steps):
        T = 1.2  # temperature
        if temperature:
            particles += np.sqrt(2*eta*T)*np.random.randn(N_samples, N_particles, 2)

        if border:
            particles += eta*drop.F(particles)

        distances = interactions.get_distance_matrix(particles)

        if (norm(distances)> 10).any():
            raise Exception("A particle evaporated!!")

        if interaction:
            final_force = eta*interactions.F(distances)

            if np.isnan(final_force).any():
                raise Exception("There is a NaN in the force!!")

            final_force_norm = np.sqrt(np.sum(final_force**2, -1, keepdims=True))
            
            # Clip maximum force to prevent numerical explosion when
            # two particles are too close
            final_force_clip = np.clip(final_force_norm, -10e-3, 10e-3)
            particles -= final_force*final_force_clip/(final_force_norm + 10e-50)

        if plot:
            # potentials.append(interactions.V(distances)[0])
            if i % int(N_steps/10) == 0:
                plot_drop(0, particles, drop, interactions, show3d=True)

    return particles


#######################################
# Some useful plotting functions

def plot_drop(i, particles, drop, interactions, ax=None, color="blue", show3d=False, save_file=""):
    """Plot a single sample

    Args:
        i (int): index of the sample in the 'particles' array 
        particles (array_like): (shape=[N_samples, N_particles, 2])
        drop (lj.Drop): drop parameters
        interactions (lj.interaction): interaction parameters
        ax (plt.figure, optional): where to plot. Defaults to None.
        color (str, optional): [description]. Defaults to "blue".
        show3d (bool, optional): enable 3d mode (paper quality). Defaults to False.
        save_file (str, optional): Path to save the plot. Defaults to "".
    """

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
    scale_scatter = (interactions.d_coll/2)*1.5
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

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis('off')
    if save_file != "":
        plt.savefig(save_file)

    if do_plt_show:
        plt.show()


def plot_pars_feature(feature, deltaR, theta, labels=None, skips=5, save_path=None):
    """Plot associated feature value for corresponding to each original deltaR and theta

    Args:
        feature (array_like): [N_samples, 2] low-dimensional feature
        deltaR (array_like): [N_samples] deformation of the drop
        theta (array_like): [N_samples] orientation of the drop
        labels (list, optional): [2] Labels of the axes. Defaults to None.
        skips (int, optional):(1/skips) is the fraction of samples to plot. Defaults to 5.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    if labels is None:
        labels = ["Feature 1", "Feature 2"]

    plt.figure(dpi=100, figsize=[12, 7])
    plt.subplot(1, 2, 1)
    plt.xlabel(r"$\Delta R$")
    plt.ylabel(r"$\theta$")
    plt.title(labels[0])
    plt.scatter(deltaR.flatten()[::skips], theta.flatten()[::skips], c=feature[:, 0][::skips], s=0.5)

    plt.subplot(1, 2, 2)
    plt.xlabel(r"$\Delta R$")
    plt.ylabel(r"$\theta$")
    plt.title(labels[1])
    plt.scatter(deltaR.flatten()[::skips], theta.flatten()[::skips], c=feature[:, 1][::skips], s=0.5)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_feature_pars(feature, deltaR, theta, labels=None, skips=5, alpha=0.01, save_path=None):
    """Plot original deltaR and theta encoding associated to given each given (feature1, feature2) point in feature space

    Args:
        feature (array_like): [N_samples, 2] low-dimensional feature
        deltaR (array_like): [N_samples] deformation of the drop
        theta (array_like): [N_samples] orientation of the drop
        labels (list, optional): [2] Labels of the axes. Defaults to None.
        skips (int, optional):(1/skips) is the fraction of samples to plot. Defaults to 5.
        alpha (float, optional): Opacity of the points in the plot. Defaults to 0.01.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    if labels is None:
        labels = ["Feature 1", "Feature 2"]

    plt.figure(figsize=[11, 5], dpi=150)
    plt.subplot(1, 2, 1)
    plt.title(r"$\Delta R$")
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.scatter(feature[:, 0][::skips], feature[:, 1][::skips], s=0.5, c=deltaR.flatten()[::skips], alpha=alpha)
    plt.colorbar()
    plt.gca().set_aspect("equal")

    plt.subplot(1, 2, 2)
    plt.title(r"$\theta$")
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.scatter(feature[:, 0][::skips], feature[:, 1][::skips], s=0.5, c=theta.flatten()[::skips], alpha=alpha)
    plt.colorbar()
    plt.gca().set_aspect("equal")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_feature_hists(feature, labels=None, save_path=None):
    """Feature distribution

    Args:
        feature (array_like): [N_samples, 2] low-dimensional feature
        labels (list, optional): [2] Labels of the axes. Defaults to None.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    if labels is None:
        labels = ["Feature 1", "Feature 2"]
    plt.figure(dpi=50, figsize=[25, 10])
    plt.subplot(1, 3, 1)
    plt.title("Distribution of " + labels[0])
    plt.hist(feature[:, 0], 30)

    plt.subplot(1, 3, 2)
    plt.title("Distribution of " + labels[1])
    plt.hist(feature[:, 1], 30)

    plt.subplot(1, 3, 3)
    plt.gca().set_aspect("equal")
    plt.title("Correlation of " + labels[0] + " " + labels[1])
    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.01)
    if save_path:
        plt.savefig(save_path)
    plt.show()
