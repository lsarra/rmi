'''
    Script to join outputs of parallel runs on the zeropoint cluster.
    Run this file in the outputs folder to join outputs of run_lennard.py
    
    version 20200428
    
    join_lennard_outputs.py

'''
import os  # NOQA
import sys  # NOQA
sys.path.insert(0, os.path.abspath('..'))  # NOQA


from glob import glob
import argparse
import pickle

import numpy as np
import rmi.examples.lennardjones as lj


def read_arrays(path):
    files = sorted(glob(path))
    return [np.load(f, allow_pickle=True) for f in files]


def read_files(path):
    files = sorted(glob(path))
    fls = []
    for f in files:
        with open(f, 'rb') as pickle_file:
            fls.append(pickle.load(pickle_file))
    return fls


def read_all_files(j, label):
    particles = read_arrays('../outputs/particles_' + label + str(j)+'*.npy')
    drops = read_files('../outputs/drop_' + label + str(j)+'*')
    interactions = read_files('../outputs/interactions_'+label+str(j)+'*')
    return particles, drops, interactions


# Read the command line arguments
parser = argparse.ArgumentParser(
    description="Join sample files generated with parallel runs",
    epilog="Results will be saved in files with the LABEL tag in the 'outputs/' folder.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-label", type=str, required=True,
                    help="Label associated to the files to join")
args = parser.parse_args()


# for j in range(10):
j = ""
label = args.label
part, drp, intr = read_all_files(j, label)

thetas = []
deltaRs = []

for i in range(len(drp)):
    thetas.append(drp[i].theta)
    deltaRs.append(drp[i].deltaR)

all_theta = np.array(thetas).reshape([-1, 1, 1])
all_deltaRs = np.array(deltaRs).reshape([-1, 1, 1])

water = lj.Drop(drp[0].R, all_deltaRs, all_theta, drp[0].A_conf)
interactions = intr[0]

particles = np.array([part])
particles = particles.reshape([-1, particles.shape[-2], particles.shape[-1]])

np.save("../outputs/PARTICLES_"+label + str(j), particles)
with open("../outputs/DROP_"+label+str(j), 'wb') as pickle_file:
    pickle.dump(water, pickle_file)
with open("../outputs/INTERACTIONS_" + label + str(j), 'wb') as pickle_file:
    pickle.dump(interactions, pickle_file)
