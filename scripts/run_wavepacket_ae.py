#!/usr/bin/env python

import os  # NOQA
import sys  # NOQA
sys.path.insert(0, os.path.abspath('..'))  # NOQA

from rmi.pca import pca
import rmi.examples.wavepacket as wp
import rmi.features as f
import rmi.estimation as inf
import rmi.neuralnets as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import argparse


class args:
    N_pixels = 100
    noise = 0.1
    width = 9.
    pos_range = [30, 70]
    neurons = 70
    N_train = 15000
    batchsize = 700
    eta = 0.0005
    output = "none"


# Read the command line arguments
parser = argparse.ArgumentParser(
    description="Example implementation of Regularized Mutual Information Feature Selector on a solid drop.",
    epilog="Results will be saved in files with the OUTPUT tag in the 'outputs/' folder.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-N_pixels", type=int, default=args.N_pixels,
                    help="Number of points in each sample")
parser.add_argument("-noise", type=float, default=args.noise,
                    help="Background noise strength")
parser.add_argument("-width", type=float, default=args.width,
                    help="width of the packet")
parser.add_argument("-pos_range", type=float, nargs="+", default=args.pos_range,
                    help="Range in which to extract the center of the packet")
parser.add_argument("-neurons", type=int, default=args.neurons,
                    help="Number of neurons of the neural network")
parser.add_argument("-N_train", type=int, default=args.N_train,
                    help="Number of training steps")
parser.add_argument("-batchsize", type=int, default=args.batchsize,
                    help="Samples in each batch")
parser.add_argument("-eta", type=float, default=args.eta,
                    help="Learning rate of the Adam algorithm to train the neural network")
parser.add_argument("-output", default=args.output,
                    help="String to append to the output")

try:
    args = parser.parse_args()
except SystemExit as e:
    print("Running from interactive session. Loading default parameters")


H_binsize = 180
H_kernel_size = 1

n_out = 1

if args.output != "none":
    path = "../models/wavepacket_ae_"+args.output
else:
    path = None


def get_batch(N_batch): return wp.produce_Wave_Packet(n_pixels=args.N_pixels,
                                                      n_samples=N_batch,
                                                      width=args.width,
                                                      noise=args.noise,
                                                      pos_range=args.pos_range)


encoder = nn.RMIOptimizer(H_binsize,
                          H_kernel_size, layers=[
                              nn.K.layers.Dense(args.neurons, activation="tanh",
                                                input_shape=(args.N_pixels,)),
                              nn.K.layers.Dense(args.neurons, activation="tanh"),
                              nn.K.layers.Dense(n_out)
                          ])

decoder = nn.K.Sequential(layers=[
    nn.K.layers.Dense(args.neurons, activation="relu",
                      input_shape=(n_out,)),
    nn.K.layers.Dense(args.neurons, activation="relu"),
    nn.K.layers.Dense(args.N_pixels)
])

autoencoder = nn.Supervised(cost="mse_contract", layers=[encoder, decoder])
autoencoder.compile(optimizer=nn.tf.optimizers.Adam(args.eta))

ae_net = nn.Net(autoencoder,
                mode="w",
                path=path)


enc_net = nn.Net(encoder)


ae_net.fit_generator(lambda: get_batch(args.batchsize)[0], args.N_train)
ae_net.plot_history(save_file="../logs/wavepacket_ae_"+args.output+".png")


samples, last_pos = wp.produce_Wave_Packet(n_pixels=args.N_pixels,
                                           n_samples=30000,
                                           width=args.width,
                                           noise=args.noise,
                                           pos_range=args.pos_range)

print("Autoencoder Network")
f_ae = encoder(samples).numpy()
wp.plot_sorted_feature(samples, f_ae, save_path="../logs/wp_comp_ae_" + args.output)

x_batch = get_batch(200)[0]
y_batch = autoencoder.predict_on_batch(x_batch)
wp.plot(x_batch, save_path="../logs/wp_" + args.output + "_x.png")
wp.plot(y_batch, save_path="../logs/wp_" + args.output + "_y.png")
