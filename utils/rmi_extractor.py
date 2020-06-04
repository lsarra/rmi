'''
rmi_extractor.py

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

Renormalized Mutual Information - Numerical Feature Extraction with TensorFlow

This package extracts optimal features by maximizing renormalized mutual information with input distribution. Since we need to estimate P(y) (distribution of the feature), this algorithm only works for a low-dimensional input space (in particular 2d in this case).

Usage:
- from rmi_extractor import *
- feature_extractor = FeatureExtractor()
- use the build() function and provide the required parameters
- train_step() as many times as required
- look at the costs array
- get the feature()

The function feature_gaugeFixed() rescales the feature to a uniform interval
'''

import numpy as np
import matplotlib.pyplot as plt
# if using tensorflow v2
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as K
# if using tensorflow v1:
# import tensorflow as tf
# import tensorflow.keras as K

class FeatureExtractor:
    # exp decay= decay steps, decay rate
    def build(self, Px, dx, X, N_neurons=180, eta=0.01, exp_decay=[50, 0.9]):
        '''
            Set up the Feature Extractor. Following parameters are required:
            - Px: distribution of the x, with shape [N, N]. N will be used as number of points in each side of the lattice
            - dx: discretization size of the lattice sum Px dx**2 = 1 in this 2d case
            - X : points of the lattice (usually meshgrid). They must be the same on which Px has been calculated
            - N_neurons of the neural network (same in each layer)
            - eta : learning rate
            - exp_decay = [decay rate, decay step] for an exponentially decaying learning rate
            In particular decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)

        '''
        self.xdelta = dx
        self.x_points = X

        self.P_x = Px
        self.N = np.shape(Px)[0]

        self.P_x_short = self.P_x.reshape([1, self.N*self.N])

        self.Ny = self.N
        self.eta = eta
        self.neurons_feature = N_neurons

        self.graph = tf.Graph()

        with self.graph.as_default():
            #############################
            # Define the input placeholders and the feature neural network
            self.tf_x = tf.placeholder(tf.float32, shape=[2, None])
            self.tf_i = tf.placeholder(tf.float32)
            self.tf_a = tf.placeholder(tf.float32)

            self.tf_theta_input = K.layers.Input(shape=(2,))
            self.tf_theta_layers = self.tf_theta_input
            self.tf_theta_layers = K.layers.Dense(self.neurons_feature,
                                                  input_shape=(2,),
                                                  activation=K.layers.ReLU(),
                                                  kernel_initializer=tf.random_normal_initializer(),
                                                  bias_initializer=tf.random_normal_initializer())(self.tf_theta_layers)
            self.tf_theta_layers = K.layers.Dense(self.neurons_feature,
                                                  activation=K.layers.ReLU(),
                                                  kernel_initializer=tf.initializers.glorot_normal(),
                                                  bias_initializer=tf.random_normal_initializer())(self.tf_theta_layers)
            self.tf_theta_layers = K.layers.Dense(self.neurons_feature,
                                                  activation=K.activations.tanh,
                                                  kernel_initializer=tf.initializers.glorot_normal(),
                                                  bias_initializer=tf.random_normal_initializer())(self.tf_theta_layers)
            self.tf_theta_layers = K.layers.Dense(1,
                                                  kernel_initializer=tf.initializers.glorot_normal(),
                                                  bias_initializer=tf.random_normal_initializer())(self.tf_theta_layers)
            self.tf_theta_net = K.Model(self.tf_theta_input,
                                        self.tf_theta_layers)

            self.tf_f = self.tf_a * tf.reshape(self.tf_theta_net(tf.transpose(self.tf_x)),
                                               [1, -1])

            #############################
            # Regularizing Term
            self.tf_grads_f = tf.gradients(self.tf_f, self.tf_x)[0]
            self.tf_norm2_grad_f = tf.reduce_sum(self.tf_grads_f**2, 0)
            self.tf_term1_local = -0.5 * (
                tf.log(self.tf_norm2_grad_f)*self.P_x_short*self.xdelta**2)
            self.tf_term1 = -0.5 * tf.reduce_sum(
                tf.log(self.tf_norm2_grad_f) * self.P_x_short*self.xdelta**2)

            #############################
            # Entropy Term

            # Define current range of the feature (in which to approximate Py)
            self.tf_y_min = tf.reduce_min(self.tf_f)
            self.tf_y_max = tf.reduce_max(self.tf_f)
            self.tf_ydelta = tf.stop_gradient(
                (self.tf_y_max-self.tf_y_min)/(self.Ny-1))
            self.tf_y_linspace = tf.reshape(tf.stop_gradient(tf.linspace(self.tf_y_min, self.tf_y_max, self.Ny)),
                                            [self.Ny, 1])

            # Define a triangular histogram (so that it is differentiable)
            self.tf_y_mask_left = tf.logical_and((self.tf_y_linspace - self.tf_ydelta < self.tf_f),
                                                 (self.tf_y_linspace > self.tf_f))
            self.tf_y_mask_right = tf.logical_and((self.tf_y_linspace <= self.tf_f),
                                                  (self.tf_y_linspace + self.tf_ydelta > self.tf_f))

            self.tf_y_line_left = (
                1/self.tf_ydelta + 1/self.tf_ydelta**2*(self.tf_f-self.tf_y_linspace))
            self.tf_y_line_right = (
                1/self.tf_ydelta - 1/self.tf_ydelta**2*(self.tf_f-self.tf_y_linspace))

            self.tf_ydelta_left = self.tf_y_line_left * tf.stop_gradient(tf.cast(self.tf_y_mask_left,
                                                                                 tf.float32))
            self.tf_ydelta_right = self.tf_y_line_right * tf.stop_gradient(tf.cast(self.tf_y_mask_right,
                                                                                   tf.float32))

            # Approximate the distribution of the feature through a differentiable histogram
            self.tf_P_y = tf.reduce_sum((self.tf_ydelta_left+self.tf_ydelta_right)*self.P_x_short*self.xdelta**2,
                                        1)
            # Calculate the Entropy of the feature
            self.tf_H_y = - tf.reduce_sum(
                self.tf_P_y*tf.log(self.tf_P_y))*self.tf_ydelta

            #############################
            # Renormalized Mutual Information and training methods
            self.tf_cost = self.tf_term1 + self.tf_H_y

            # Optimizer (with exponential decaying learning rate)
            self.tf_optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=tf.train.exponential_decay(self.eta, self.tf_i, exp_decay[0], exp_decay[1]))

            # Gradients of the cost function
            self.tf_grad_cost = self.tf_optimizer.compute_gradients(
                -self.tf_cost,
                self.tf_theta_net.trainable_variables)

            # Train step
            self.tf_train_step = self.tf_optimizer.apply_gradients(
                self.tf_grad_cost)

            # Initialize the neural network
            self.tf_init_op = tf.global_variables_initializer()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.tf_init_op)
        self.costs = []

    def summary(self):
        '''
        Prints the summary of the neural network
        '''
        print("\nFeature Extractor:")
        self.tf_theta_net.summary()

    def train_step(self, j=1):
        '''
        Train the neural network on the grid X. 
        It saves the new value of the cost function in the costs array.
        - provide j= current # training step to implement an exponential decay of the learning rate.
        '''
        feed = {self.tf_x: self.x_points, self.tf_i: j, self.tf_a: 1}
        #feed = {tf_x:sp.sample(batchsize).T}
        _, c = self.sess.run([self.tf_train_step, self.tf_cost], feed)
        self.costs.append(c)

    def feature_at(self, x_points, rescale=1):
        feed = {self.tf_x: x_points.T, self.tf_a: rescale}
        return self.sess.run(self.tf_f, feed)

    def feature(self, rescale=1):
        '''
        Returns the feature f calculated on the given x_points.
        If no points are provided, the feature is calculated on the grid X
        - If rescale is provided, feature is multiplied by rescale
        '''
        x_points = self.x_points

        feed = {self.tf_x: x_points, self.tf_a: rescale}
        return self.sess.run(self.tf_f, feed).reshape([self.N, self.N])

    # def feature_gaugeFixed(self, rescale=1):
    #     '''
    #     Fix the gauge of the feature. In particular, we choose H(y=f(x)) to have a uniform distribution.
    #     This allows to numerically compare different feature. The only remaining ambiguity is the sign of the gradient of the feature (increasing feature or decreasing feature). For example, this can be fixed by flipping all decreasing feature (choose rescale = -1 if feature is decreasing, otherwise rescale = 1)
    #     '''
    #     import scipy as sc
    #     from scipy import interpolate

    #     f = self.feature(rescale=rescale)
    #     feed = {self.tf_x: self.x_points, self.tf_a: rescale}
    #     ydelta = self.sess.run(self.tf_ydelta, feed)
    #     Py = self.sess.run(self.tf_P_y, feed)
    #     # It is very easy to obtain the function to transform the feature to have uniform distribution
    #     # Indeed, we know that the cumulative of the distribution gives the right result.
    #     # It is crucial to multiply by ydelta because ydelta is not fixed, but different each time (and different from 1)
    #     G = np.cumsum(Py)*ydelta
    #     y_linspace = self.sess.run(self.tf_y_linspace, feed).flatten()

    #     f_norm = sc.interpolate.CubicSpline(y_linspace, G)(f)
    #     return f_norm

    def feature_gaugeFixed(self, rescale=1):
        '''
        Fix the gauge of the feature. In particular, we choose H(y=f(x)) to have a uniform distribution.
        This allows to numerically compare different feature. The only remaining ambiguity is the sign of the gradient of the feature (increasing feature or decreasing feature). For example, this can be fixed by flipping all decreasing feature (choose rescale = -1 if feature is decreasing, otherwise rescale = 1)
        '''
        import scipy as sc
        from scipy import interpolate
        f = self.feature(rescale=rescale)

        ff = f.flatten()
        ww = self.P_x.flatten()
        Py, delt = np.histogram(ff, 10000, weights=ww, density=True)
        ydelta = delt[1]-delt[0]
        y_linspace = delt[1:]-ydelta/2
        G = np.cumsum(Py)*ydelta
        ffnorm = sc.interpolate.CubicSpline(y_linspace, G)(f)
        return ffnorm

    def cost(self):
        '''
        Evaluates the renormalized mutual information of x with the feature y.
        '''
        feed = {self.tf_x: self.x_points, self.tf_a: 1}
        return self.sess.run(self.tf_cost, feed)
