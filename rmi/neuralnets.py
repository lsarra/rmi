# neuralnets.py

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

import tensorflow as tf
import tensorflow.keras as K

from tqdm import trange


class Supervised(K.Sequential):
    """Keras model that implements custom cust functions and training loop
    """
    def __init__(self, cost,  **kwargs):
        """Initialize the class (in addition to common Keras model arguments, 
        like `layers`)

        Args:
            cost (string): cost function of the model. Possible choices are 
                - mean squared error ("mse")
                - modified mean squared error - use for liquid drop example("mse_lj")
                - contractive mean squared error - in case of a contractive autoencoder ("mse_contract")
        """
        super(Supervised, self).__init__(**kwargs)
        self.cost = cost
        if cost == "mse":
            self.tf_calcCost = self.tf_mse
        elif cost == "mse_lj":
            self.tf_calcCost = self.tf_mse_lj
        elif cost == "mse_contract":
            self.tf_calcCost = self.tf_mse_contract

    @tf.function
    def tf_mse(self, tf_f, tf_labels): 
        """ mse cost function

        Args:
            tf_f (tensor_like): [N_samples, N_y] output of the network
            tf_labels (tf_like): [N_samples, N_y] true value

        Returns:
            list: [1] value of the cost
        """
        return [tf.reduce_mean((tf_f-tf_labels)**2)]

    @tf.function
    def tf_mse_contract(self, tf_labels, tf_x): 
        """Contractive mse cost function

        Like mean squared error but with an additional penalty.

        It should be applied when the Supervised class is actually an Autoencoder,
        its first layer should be the encoder (we calculate the penalty on this layer)
        and the second layer is the decoder.

        Args:
            tf_labels (tensor_like):  [N_samples, N_x] output of the network
            tf_x (tensor_like): [N_samples, N_x]

        Returns:
            list: [2] value of the cost, value of the penalty

        """
        # tf_grads has shape [Ny, Nx, Nsamples]
        _, tf_grads = self.layers[0].tf_calcFeature(tf_x)
        # Frobenius norm of the gradients, then average over samples
        tf_price = 0.01
        tf_penalty = tf.reduce_mean(tf.math.reduce_euclidean_norm(tf_grads, (0, 1)))
        return [tf.reduce_mean((self(tf_x)-tf_labels)**2) + tf_price*tf_penalty,
                tf_penalty]

    @tf.function
    def tf_mse_lj(self, tf_f, tf_labels):  
        """Custom mean squared error for Liquid drop example

        Cost function that takes into account that the orientation
        of the drop is an angle

        To predict the deformation and orientation of the drop, 
        we employ 3 outputs:
        - dr (deformation)
        - cos(th) (cosine of orientation)
        - sin(th) (sine of orientation)

        Args:
            tf_f (tensor_like): [N_samples, 3] output of the network
            tf_labels (tensor_like): [N_samples, 3] predictions

        Returns:
            (tf_float): total cost
            (tf_float): delta_R error
            (tf_float): cos(th) error
            (tf_float): sin(th) error
        """
        tf_deltaR = (tf_f - tf_labels)[:, 0]
        tf_deltaCOS = tf_f[:, 1] - tf_labels[:, 1]**2 + tf_labels[:, 2]**2
        tf_deltaSIN = tf_f[:, 2] - 2 * tf_labels[:, 1]*tf_labels[:, 2]

        tf_mse_deltaR = (tf_deltaR**2)
        tf_mse_deltaCOS = (tf_labels[:, 0]**2*tf_deltaCOS**2)
        tf_mse_deltaSIN = (tf_labels[:, 0]**2*tf_deltaSIN**2)

        return (tf.reduce_mean(tf_mse_deltaR + tf_mse_deltaCOS + tf_mse_deltaSIN),
                tf.reduce_mean(tf_mse_deltaR),
                tf.reduce_mean(tf_mse_deltaCOS),
                tf.reduce_mean(tf_mse_deltaSIN))

    @tf.function
    def train_step(self, tf_x, tf_labels):
        """Perform a single train step of the network

        Args:
            tf_x (tensor_like): [N_samples, N_x] network input
            tf_labels (tensor_like): [N_samples, N_y] network output

        Returns:
            tf_costs (list): loss and metrics list
        """
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            tf_f = self(tf_x)
            if self.cost == "mse_contract":
                tf_costs = self.tf_calcCost(tf_labels, tf_x)
            else:
                tf_costs = self.tf_calcCost(tf_f, tf_labels)

            tf_loss = tf_costs[0]
        tf_grads = tape.gradient(tf_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(tf_grads, self.trainable_variables))
        return tf_costs


class RMIOptimizer(K.Sequential):
    """Keras model that implements Renormalized Mutual Information cost function
    """
    def __init__(self, 
                 H_nbins=180,
                 H_kernel_size=1, 
                 coeff_gauss=0,
                 coeff_var=1,
                 reg_amplitude=0, 
                 reg_decay=None, **kwargs):
        """Class initialization

        Args:
            H_nbins (int, optional): Number of bins to estimate feature entropy. Defaults to 180.
            H_kernel_size (int, optional): Size of the blob around each point 
                (to make the histogram differentiable, when estimating entropy). Defaults to 1.
            coeff_gauss (int, optional): Strength of Gaussian-distributed-output constraint. Defaults to 0.
            coeff_var (int, optional): Variance of the output Gaussian distribution. Defaults to 1.
            reg_amplitude (int, optional): Strength of the feature gradient regularization. Defaults to 0.
            reg_decay ([type], optional): Decay constant of the feature gradient regularization. Defaults to None.

        """
        super(RMIOptimizer, self).__init__(**kwargs)
        self.reg_amplitude = reg_amplitude
        self.reg_decay = reg_decay
        self.coeff_H = coeff_gauss
        self.coeff_var = coeff_var
        self.H_nbins = H_nbins
        self.H_kernel_size = H_kernel_size

        # Define function at runtime to use the right precision signature
        @tf.function(input_signature=[tf.TensorSpec((None, None), K.backend.floatx())])
        def tf_calcFeature(tf_x):
            """Calculate the feature and its gradient.

            It corresponds to applying the neural network to the argument
            and calculating the gradient of the output with respect to the inputs.

            Args:
                tf_x (tensor_like): [N_samples, N_x] input of the neural network

            Returns:
                tf_f (tensor_like): [N_y, N_samples] output of the neural network
                tf_grads_f (tensor_like): [N_y, N_x, N_samples]
            """
            with tf.GradientTape() as tape:
                tape.watch(tf_x)
                tf_f = self(tf_x)  # should probably specify when Training=true
            tf_grads_f = tape.batch_jacobian(tf_f, tf_x)
            return tf.transpose(tf_f), tf.transpose(tf_grads_f, [1, 2, 0])
        self.tf_calcFeature = tf_calcFeature


    @tf.function
    def tf_math_erf(self,x):
        """Clipped error function

        To avoid numerical inaccuracies due to finite precision,
        it is helpful to cut the error function for very large values

        Args:
            x (tensor_like): input 

        Returns:
            (tensor_like): output
        """
        return  tf.clip_by_value(tf.math.erf(x),-0.97,0.97)/0.97

    @tf.function
    def tf_calcProbabilityDistribution(self, tf_f):
        """Estimate feature density distribution

        We estimate the feature probability distribution
        with a differentiable histogram.

        NOTE: Unfortunately, at the moment this is implemented only for
        a one or two dimensional feature. 
        Generalizations to higher-dimensional features would be unfeasible
        to do with a histogram. 
        An alternative and more efficient technique must be implemented.

        Args:
            tf_f (tensor_like): [N_y, N_samples]

        Returns:
            (tensor_like): [H_nbins] or [H_nbins,H_nbins] according to N_y = 1 or N_y = 2

        Raises:
            NotImplementedError: feature is more than two-dimensional
        """
    
        if tf_f.get_shape()[0] == 1:
            tf_y_minimum = tf.reduce_min(tf_f)
            tf_y_maximum = tf.reduce_max(tf_f)
            tf_ydlt = tf.stop_gradient((tf_y_maximum-tf_y_minimum)/tf.cast(self.H_nbins-1, K.backend.floatx()))

            # Define current range of the feature (in which to approximate Py)
            # Actual histogram bounds
            tf_y_min = tf_y_minimum - 3*tf_ydlt
            tf_y_max = tf_y_maximum + 3*tf_ydlt

            tf_y_linspace = tf.reshape(
                tf.stop_gradient(tf.linspace(tf_y_min, tf_y_max, self.H_nbins)),
                [self.H_nbins, 1])
            tf_ydelta = tf.stop_gradient((tf_y_max-tf_y_min)/tf.cast(self.H_nbins-1, K.backend.floatx()))
            tf_y_linspace_plus = tf.stop_gradient(tf_y_linspace + tf_ydelta)

            # Put a gaussian around each sample and integrate it in each bin
            tf_histY = 0.5*(
                self.tf_math_erf((tf_y_linspace_plus - tf_f)/(np.sqrt(2)*tf_ydelta*self.H_kernel_size))
                - self.tf_math_erf((tf_y_linspace - tf_f)/(np.sqrt(2)*tf_ydelta*self.H_kernel_size))
            )
            return tf.reduce_mean(tf_histY, -1)/(tf_ydelta), tf_ydelta

        elif tf_f.get_shape()[0] == 2:
            # Just temporary
            tf_y_minimum = tf.reduce_min(tf_f, 1)
            tf_y_maximum = tf.reduce_max(tf_f, 1)
            tf_ydlt = tf.stop_gradient((tf_y_maximum-tf_y_minimum)/tf.cast(self.H_nbins-1, K.backend.floatx()))

            # Define current range of the feature (in which to approximate Py)
            # Actual histogram bounds
            tf_y_min = tf_y_minimum - 3*tf_ydlt
            tf_y_max = tf_y_maximum + 3*tf_ydlt

            tf_ydelta = tf.stop_gradient((tf_y_max-tf_y_min)/tf.cast(self.H_nbins-1, K.backend.floatx()))

            tf_y1_linspace = tf.reshape(tf.stop_gradient(tf.linspace(tf_y_min[0], tf_y_max[0], self.H_nbins)), [self.H_nbins, 1, 1])
            tf_y2_linspace = tf.reshape(tf.stop_gradient(tf.linspace(tf_y_min[1], tf_y_max[1], self.H_nbins)), [1, self.H_nbins, 1])

            tf_y1_linspace_plus = tf.stop_gradient(tf_y1_linspace + tf_ydelta[0])
            tf_y2_linspace_plus = tf.stop_gradient(tf_y2_linspace + tf_ydelta[1])

            # Put a gaussian around each sample and integrate it in each bin
            tf_histY1 = 0.5*(self.tf_math_erf((tf_y1_linspace_plus - tf_f[0])/(np.sqrt(2)*tf_ydelta[0]*self.H_kernel_size))
                            - self.tf_math_erf((tf_y1_linspace - tf_f[0])/(np.sqrt(2)*tf_ydelta[0]*self.H_kernel_size)))

            tf_histY2 = 0.5*(self.tf_math_erf((tf_y2_linspace_plus - tf_f[1])/(np.sqrt(2)*tf_ydelta[1]*self.H_kernel_size))
                            - self.tf_math_erf((tf_y2_linspace - tf_f[1])/(np.sqrt(2)*tf_ydelta[1]*self.H_kernel_size)))

            tf_histY = (tf_histY1*tf_histY2)

            # Approximate the distribution of the feature through a differentiable histogram
            return tf.reduce_mean(tf_histY, -1)/(tf_ydelta[0]*tf_ydelta[1]), tf_ydelta
        else:
            raise NotImplementedError("The method is not yet implemented for high-dimensional features (N_y > 2).")

    @tf.function
    def tf_calcEntropy(self, tf_f):
        """Estimate feature entropy

        Args:
            tf_f (tensor_like): [N_y, N_samples] output of the neural network

        Returns:
            (tf_float): calculated entropy of the feature distribution
        """
        tf_P_y, tf_ydelta = self.tf_calcProbabilityDistribution(tf_f)
        return  - tf.reduce_sum(tf.math.xlogy(tf_P_y, tf_P_y + K.backend.epsilon()))*tf.reduce_prod(tf_ydelta)

    @tf.function
    def tf_calcFterm(self, tf_grads_f):
        """Calculate the renormalizing term of Renormalized Mutual Information

        Fterm = -1/2 < log det gradF . gradF >_x

        Args:
            tf_grads_f (tensor_like): [N_y, N_x, N_samples] gradient of the feature with respect to inputs

        Returns:
            (tf_float): calculated value
        """
        tf_gradmat = tf.transpose(tf_grads_f, (2, 0, 1))@tf.transpose(tf_grads_f, (2, 1, 0))

        # Numerically, it is helpful to add a small regularization (we chose 1e-3) to prevent the  
        # logarithm to divergence. This helps to compensate the inaccuracies of entropy due to finite
        # sampling.
        tf_term = -0.5*tf.reduce_mean(tf.math.log(tf.linalg.det(tf_gradmat) + 1e-3))
        return tf_term

    @tf.function
    def tf_calcRenormalizedMutualInformation(self, tf_f, tf_grads_f):
        """Estimate Renormalized Mutual Information 

        Calculate RMI(x, f(x)) = H(f) + Fterm(gradF)

        Please NOTE: at the moment only RMI of at most 2d features can be calculated

        Args:
            tf_f (tensor_like): [N_y, N_samples] feature
            tf_grads_f (tensor_like): [N_y, N_x, N_samples] feature gradient

        Returns:
            (tf_float): calculated renormalized mutual information
        """
        tf_H = self.tf_calcEntropy(tf_f)
        tf_Fterm = self.tf_calcFterm(tf_grads_f)
        tf_RMI = tf_H + tf_Fterm
        return tf_RMI

    @tf.function
    def tf_reg_varF(self, tf_f):
        """Feature Variance

        Args:
            tf_f (tensor_like): [N_y, N_samples] feature

        Returns:
            (tf_float): calculated variance
        """
        return 0.5*tf.reduce_mean(tf.reduce_sum(tf_f**2, 0))


    @tf.function
    def tf_reg_gradF(self, tf_grads_f):
        """Feature Gradient regularization

        Args:
            tf_grads_f (tensor_like): [N_y, N_x, N_samples] feature gradient

        Returns:
            (tf_float): calculated term
        """
        # This has shape [Ny,N_samples]
        tf_norm2_grad_f = tf.math.reduce_euclidean_norm(tf_grads_f, 1)**2
        # here we average over the samples and on the two components
        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf_norm2_grad_f + K.backend.epsilon(), 0)))

    @tf.function
    def tf_calcCost(self, tf_f, tf_grads_f, i):
        """Calculate RMI loss

        Args:
            tf_f (tensor_like): [N_y, N_samples] feature
            tf_grads_f (tensor_like): [N_y, N_x, N_samples] feature gradient
            i (int): current training step (used for regularization decay)

        Returns:
            (list): loss and metrics
        """
        tf_H = self.tf_calcEntropy(tf_f)
        tf_Fterm = self.tf_calcFterm(tf_grads_f)
        tf_RMI = tf_H + tf_Fterm

        tf_rVar = self.tf_reg_varF(tf_f)

        tf_rgradF = self.tf_reg_gradF(tf_grads_f)
        if self.reg_decay is None:
            tf_exp_decay = 1
        else: tf_exp_decay = self.reg_amplitude*tf.exp(-i/self.reg_decay)

        tf_cost = -tf_RMI + self.coeff_H*(1/self.coeff_var*tf_rVar - tf_H) + tf_exp_decay*tf_rgradF 
        return tf_cost, tf_RMI, tf_Fterm, tf_H, tf_rVar, tf_rgradF, tf_exp_decay, tf_cost

    @tf.function
    def train_step(self, tf_x, tf_y, i=1):
        """Perform a single training step

        Args:
            tf_x (tensor_like): [N_samples, N_x]
            tf_y : not needed, just to keep the same signature
            i (int, optional):  current training step (used for regularization decay). Defaults to 1.

        Returns:
            (list): loss and metrics
        """
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            tf_f, tf_grads_f = self.tf_calcFeature(tf_x)
            tf_costs = self.tf_calcCost(tf_f, tf_grads_f, i)
            tf_loss = tf_costs[0]
        tf_grads = tape.gradient(tf_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(tf_grads, self.trainable_variables))
        return tf_costs[1:]


# Class to add some useful functionalities
class Net():
    """Class to handle the training of a Keras model,
    automatically saving the history
    and implementing the functions to provide the associated feature.
    It also includes some useful functions to plot the history of the training.
    """

    def __init__(self, tf_net, mode="w", path=None):
        """Initialize the class

        Args:
            tf_net (K.Model): Associated Keras neural network model 
            mode (str, optional): Either write the model as a new file, overwriting it if it already exists ("w") 
                                or append the training to an existing model ("a"), specified in path. Defaults to "w".
            path (str, optional): Path to save/load the model. Defaults to None.

        Raises:
            Exception: append option can be specified only when also specifying the path from where the 
                model should be loaded.
        """
        self.net = tf_net
        self.history = []

        self.mode = mode
        self.path = path

        if self.mode == "a":
            if self.path is None:
                raise Exception("Cannot use append mode if no file path is specified")
            self.load()

    def fit_generator(self, get_batch, N_train, force_training=False, autosave=True):
        """Train the neural network

        Args:
            get_batch (function): ()=>[N_samples, N_samples] returns a batch of (input, label) that should be used for training
            N_train (int): number of training steps
            force_training (bool, optional): Whether to train the model if it has already been trained at least once. Defaults to False.
            autosave (bool, optional): Whether automatically save the model after the end of the training. Defaults to True.
        """
        # Training of RMI nets should be performed in a different way because faster...
        if len(self.history) == 0 or force_training:
            print("Starting the training of the model...")
        else:
            print("Model already trained, skipping training.")
            return

        for _ in trange(N_train):
            x_batch = get_batch()
            # If the batch function returns a list (x_batch, y_batch)
            # then consider the first as input and the second as label
            # If only an array is returned, we consider the same as both input and label
            if len(x_batch)==2:
                x_batch, y_batch = x_batch
            else:
                y_batch = x_batch

            tf_x = tf.convert_to_tensor(x_batch, dtype=K.backend.floatx())
            tf_y = tf.convert_to_tensor(y_batch, dtype=K.backend.floatx())
            tf_i = tf.convert_to_tensor(len(self.history), dtype=K.backend.floatx())
            if isinstance(self.net, RMIOptimizer):
                history = self.net.train_step(tf_x, tf_y, i=tf_i)
            else:
                history = self.net.train_step(tf_x, tf_y)

            # Save as numpy object (and not tf_tensor)
            numpy_hist = [history[i].numpy() for i in range(len(history))]
            self.history.append(numpy_hist)

        if autosave:
            self.save()

    def summary(self):
        """Table with the neural network's layout.
        """
        self.net.summary()

    def save(self, path=None):
        """Save the model (and training history) to file

        Args:
            path (str, optional): Path tho save the model and the history. Defaults to None.
        """
        
        if path is None:
            path = self.path

        if path is not None:
            self.net.save_weights(path)
            np.save(path + "_history.npy", self.history)
            print("Saved neural network weights to {}".format(path))
        else:
            print("No saving path specified. Network won't be saved to disk.")

    def load(self, path=None):
        """Load a saved model

        Args:
            path (str, optional): Path of the model. If none is specified,
                the internal path of the class will be used. Defaults to None.
        """
        if path is None:
            path = self.path
        try:
            self.net.load_weights(path)
            self.history = np.load(path + "_history.npy", allow_pickle=True).tolist()
            print("Loaded neural network weights from {}".format(path))
        except Exception as e:
            print("No saved network found. Network needs to be trained from scratch\n")
            print(e)

    def get_feature_and_grad(self, samples):
        """Calculate feature and gradient

        Args:
            samples (array_like): [N_samples, N_x] network input  

        Raises:
            Exception: when associated net is not a RMIOptimizer model
                Only in that case the gradient of the feature can be calculated.

        Returns:
            feature (array_like): [N_samples, N_y] network output
            grad_feature (array_like): [N_samples, N_y, N_x] gradient of the output with respect to input
        """
        if not isinstance(self.net, RMIOptimizer):
            raise Exception("Only applies to RMI nets")
        tf_x = tf.convert_to_tensor(samples, K.backend.floatx())
        tf_f, tf_grad = self.net.tf_calcFeature(tf_x)
        return tf_f.numpy().T, tf_grad.numpy().T.swapaxes(1, 2)

    def __call__(self, samples):
        """Apply the network on the inputs

        Same as `get_feature_and_grad` but does not calculate gradients.

        Args:
            samples (array_like): [N_samples, N_x] network input  

        Returns:
            feature (array_like): [N_samples, N_y] network output
        """
        tf_x = tf.convert_to_tensor(samples, K.backend.floatx())
        tf_f = self.net(tf_x)
        return tf_f.numpy().T


    def plot_rmi_cost(self, start=0, save_file=None):
        """Plot RMI network training history

        Args:
            start (int, optional): Beginning of the plot. Defaults to 0.
            save_file (str, optional): Path to save the plot. Defaults to None.
        """
        hst = np.transpose(self.history)

        if self.net.output_shape[-1] == 1:
            sigma = 1
            H_y_target = np.log(sigma*np.sqrt(2*np.pi*np.exp(1)))
        elif self.net.output_shape[-1] == 2:
            sigma = np.eye(2)
            H_y_target = 0.5*np.log(np.linalg.det(2*np.pi*np.exp(1)*sigma))

        fig = plt.figure(figsize=[20, 18], dpi=75, constrained_layout=False)
        gs1 = fig.add_gridspec(nrows=4, ncols=2, left=0.05, right=0.48, wspace=0.05)

        fig.add_subplot(gs1[0, :])
        plt.plot(hst[0][start:])
        plt.title("Renormalized Mutual Information")
        plt.xlabel("Training step")
        plt.ylabel("RMI(x,y=f(x))")

        fig.add_subplot(gs1[1, 0])
        plt.plot(hst[1], color="orange")
        plt.title("F term")
        plt.xlabel("Training step")
        plt.ylabel("-1/2<log grad f(x)>")

        fig.add_subplot(gs1[1, 1])
        plt.plot(hst[2][start:], color="green")

        plt.hlines(H_y_target, 0, len(hst[2][start:])-1, colors="black", linestyles="dashed", alpha=0.7)
        plt.hlines(H_y_target*1.1, 0, len(hst[2][start:])-1, colors="black", linestyles="dashed", alpha=0.3)
        plt.hlines(H_y_target*0.9, 0, len(hst[2][start:])-1, colors="black", linestyles="dashed", alpha=0.3)
        plt.title("Feature Entropy")
        plt.xlabel("Training step")
        plt.ylabel("H(y)")

        fig.add_subplot(gs1[2, 0])
        plt.plot(hst[4], color="red")
        plt.title("Feature Gradient regularizer")
        plt.xlabel("Training step")
        plt.ylabel("Grad[f]")

        fig.add_subplot(gs1[2, 1])
        plt.plot(hst[3][start:], color="darkgreen")
        plt.hlines(0, 0, len(hst[3][start:])-1, colors="black", linestyles="dashed", alpha=0.7)
        plt.hlines(0.1, 0, len(hst[3][start:])-1, colors="black", linestyles="dashed", alpha=0.3)
        plt.hlines(0.2, 0, len(hst[3][start:])-1, colors="black", linestyles="dashed", alpha=0.1)
        plt.title("Feature Variance")
        plt.xlabel("Training step")
        plt.ylabel("Var[f]")

        fig.add_subplot(gs1[3, :])
        plt.plot(hst[-1][start:],label="total loss", color="black")
        plt.plot(-hst[0][start:], label="-RMI", color="blue")
        plt.plot(- self.net.coeff_H*hst[3][start:], label="var", color="darkgreen")
        plt.plot(self.net.coeff_var*hst[2][start:], label="maxH", color="green")
        plt.plot(hst[-2][start:]*hst[4][start:], label="gradF", color="red")
        # plt.plot(hst[-3][start:], label="mean", color="purple")
        plt.title("Renormalized Mutual Information - all terms")
        plt.xlabel("Training step")
        plt.ylabel("RMI(x,y=f(x))")
        plt.legend()
        plt.yscale("log")
        if save_file is not None:
            plt.savefig(save_file)

        plt.show()

    def plot_mse_lj_costs(self, start=0):
        """Plot training history of a Lennard-Jones liquid drop supervised network

        Args:
            start (int, optional): Beginning of the plot. Defaults to 0.
        """
        hst = np.transpose(self.history)

        fig = plt.figure(figsize=[20, 15], dpi=75, constrained_layout=False)
        gs1 = fig.add_gridspec(nrows=3, ncols=2, left=0.05, right=0.48, wspace=0.05)

        fig.add_subplot(gs1[0, :])
        plt.plot(hst[0][start:])
        plt.title("Overall cost function")
        plt.xlabel("Training step")
        plt.ylabel("Cost")

        fig.add_subplot(gs1[1, :])
        plt.plot(hst[1][start:], color="orange")
        plt.title(r"$\Delta R$")
        plt.xlabel("Training step")
        plt.ylabel(r"$\delta R$")

        fig.add_subplot(gs1[2, 0])
        plt.plot(hst[2][start:], color="red")
        plt.title(r"$\Delta \cos\theta$")
        plt.xlabel("Training step")
        plt.ylabel(r"$\Delta \cos\theta$")

        fig.add_subplot(gs1[2, 1])
        plt.plot(hst[3][start:], color="blue")
        plt.title(r"$\Delta \sin\theta$")
        plt.xlabel("Training step")
        plt.ylabel(r"$\Delta \sin\theta$")

        plt.show()

    def plot_mse_contractive_costs(self, start=0):
        """Plot training history of a contractive autoencoder

        Args:
            start (int, optional): Beginning of the plot. Defaults to 0.
        """
        hst = np.transpose(self.history)

        fig = plt.figure(figsize=[20, 15], dpi=75, constrained_layout=False)
        gs1 = fig.add_gridspec(nrows=3, ncols=2, left=0.05, right=0.48, wspace=0.05)

        fig.add_subplot(gs1[0, :])
        plt.plot(hst[0][start:])
        plt.title("Overall cost function")
        plt.xlabel("Training step")
        plt.ylabel("Cost")

        fig.add_subplot(gs1[1, :])
        plt.plot(hst[1][start:], color="orange")
        plt.title("$Gradient Penalty$")
        plt.xlabel("Training step")
        plt.ylabel(r"$||\nabla E(x)||$")

        plt.show()

    def plot_history(self, start=0, save_file=None, all=False):
        """Plot training history

        Args:
            start (int, optional): Beginning of the plot. Defaults to 0.
            save_file (str, optional): Path to save the plot. Defaults to None.
            all (bool, optional): plot all metrics or only loss. Defaults to False.
        """
        hst = np.transpose(self.history)

        plt.figure()
        if all:
            plt.plot(hst[:,start:])
        else: plt.plot(hst[0,start:])
        plt.show()
        if save_file is not None:
            plt.savefig(save_file)