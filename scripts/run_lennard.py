import os  # NOQA
import sys  # NOQA
sys.path.insert(0, os.path.abspath('..'))  # NOQA


import argparse
import pickle
import numpy as np
import rmi.examples.lennardjones as lj


class args:
    N_particles = 60
    N_samples = 10
    N_steps = 1000
    d_attr = 0.4
    deltaRstd = 0.8
    drop_wall = 50
    output = 'none'


# Read the command line arguments
parser = argparse.ArgumentParser(
    description="Example implementation of Regularized Mutual Information Feature Selector on a solid drop.",
    epilog="Results will be saved in files with the OUTPUT tag in the 'outputs/' folder.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-N_samples", type=int, default=args.N_samples,
                    help="Number of samples")
parser.add_argument("-N_particles", type=int, default=args.N_particles,
                    help="Number of particles")
parser.add_argument("-deltaRstd", type=float, default=args.deltaRstd,
                    help="std. of the deltaR of the ellipse (each samples has different shape according to a Gaussian distribution with this std.")
parser.add_argument("-d_attr", type=float, default=args.d_attr,
                    help="Equilibrium position of the particles")
parser.add_argument("-drop_wall", type=float, default=args.drop_wall,
                    help="Strength of the drop confinement potential")
parser.add_argument("-N_steps", type=int, default=args.N_steps,
                    help="Number of relaxation steps")
parser.add_argument("-output", default=args.output,
                    help="String to append to the output")
try:
    args = parser.parse_args()
except SystemExit as e:
    print("Running from interactive session. Loading default parameters")


theta = np.random.uniform(0, np.pi, size=(args.N_samples, 1, 1))
deltaR = np.random.uniform(0., 1, size=(args.N_samples, 1, 1))*args.deltaRstd


water = lj.Drop(R=1.0,
                deltaR=deltaR,
                theta=theta,
                ampl_confinement=args.drop_wall)

interactions = lj.Interactions(d_collision=0.06,
                               d_attraction=lj.get_packing_distance(water.R, args.N_particles)*1.1)

particles = lj.sample(args.N_samples, args.N_particles, water)
_ = lj.relax(particles, water, interactions, eta=10e-5,
             N_steps=args.N_steps)

if args.output != "none":
    np.save("outputs/particles_"+args.output, particles)
    with open("outputs/drop_"+args.output, 'wb') as pickle_file:
        pickle.dump(water, pickle_file)
    with open("outputs/interactions_"+args.output, 'wb') as pickle_file:
        pickle.dump(interactions, pickle_file)
