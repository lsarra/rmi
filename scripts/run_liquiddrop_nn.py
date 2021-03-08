
import os  # NOQA
import sys  # NOQA
sys.path.insert(0, os.path.abspath('..'))  # NOQA

from rmi.pca import pca
import rmi.examples.lennardjones as lj
import rmi.features as f
import rmi.estimation as inf
import rmi.neuralnets as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import argparse


class args:
    range = 19
    n_neurons = 800
    N_train = 5000
    batchsize = 5000
    eta = 0.0002
    reg_decay = 1000
    reg_amplitude = 15
    coeff_H = 0.05
    coeff_var = 0.05
    output = "none"


# Read the command line arguments
parser = argparse.ArgumentParser(
    description="Example implementation of Regularized Mutual Information Feature Selector on a solid drop.",
    epilog="Results will be saved in files with the OUTPUT tag in the 'outputs/' folder.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-range", type=int, default=args.range,
                    help="Subset of samples to consider (in our code, [0,20) )")
parser.add_argument("-n_neurons", type=int, default=args.n_neurons,
                    help="Number of neurons of the neural network")
parser.add_argument("-N_train", type=int, default=args.N_train,
                    help="Number of training steps")
parser.add_argument("-batchsize", type=int, default=args.batchsize,
                    help="Samples in each batch")
parser.add_argument("-eta", type=float, default=args.eta,
                    help="Learning rate of the Adam algorithm to train the neural network")
parser.add_argument("-reg_amplitude", type=float, default=args.reg_amplitude,
                    help="Learning rate of the Adam algorithm to train the neural network")
parser.add_argument("-reg_decay", type=float, default=args.reg_decay,
                    help="Learning rate of the Adam algorithm to train the neural network")
parser.add_argument("-coeff_H", type=float, default=args.coeff_H,
                    help="Learning rate of the Adam algorithm to train the neural network")
parser.add_argument("-coeff_var", type=float, default=args.coeff_var,
                    help="Learning rate of the Adam algorithm to train the neural network")
parser.add_argument("-output", default="none",
                    help="String to append to the output")

try:
    args = parser.parse_args()
except SystemExit as e:
    print("Running from interactive session. Loading default parameters")


label = "liquid-drop-60-final"

n_linsp = np.linspace(0.1, 0.8, 20)
samples, water, interactions = lj.load_dataset(label, n_linsp[args.range])

N_samples = len(samples)
n_in = np.shape(samples)[1]
print("Training dataset: %d samples" % N_samples)


N_out = 2

batchsize = args.batchsize
N_batches = int(N_samples/batchsize)-1

eta = args.eta

if args.output != "none":
    path = "../models/liquid-drop_"+args.output
else:
    path = None

H_binsize = 100
H_kernel_size = 2
n_out = 2


def get_batch(batchsize):
    random_indices = np.random.choice(np.arange(N_samples), size=batchsize)
    return samples[random_indices], None, random_indices


rmi_optimizer = nn.RMIOptimizer(H_binsize,
                                H_kernel_size,
                                coeff_gauss=args.coeff_H,
                                reg_amplitude=args.reg_amplitude,
                                reg_decay=args.reg_decay,
                                layers=[
                                    nn.K.layers.Dense(args.n_neurons, activation="relu",
                                                      input_shape=(n_in,)),
                                    nn.K.layers.Dense(n_out)
                                ])

rmi_optimizer.compile(optimizer=nn.tf.optimizers.RMSprop(eta, epsilon=1e-9))
rmi_net = nn.Net(rmi_optimizer,
                 mode="w",
                 path=path)

rmi_net.fit_generator(lambda: get_batch(args.batchsize)[0], args.N_train, force_training=True)
rmi_net.plot_rmi_cost(save_file="../logs/"+args.output+".pdf")

feature_rmi = rmi_net.get_feature_and_grad(samples)
lj.plot_pars_feature(feature_rmi[0], water.deltaR, water.theta, save_path="../logs/liquid_drop_" + args.output+"_pf.png")
lj.plot_feature_pars(feature_rmi[0],  water.deltaR, water.theta, save_path="../logs/liquid_drop_"+args.output+"_fp.png")
lj.plot_feature_hists(feature_rmi[0], save_path="../logs/liquid_drop_"+args.output+"_hist.png")
