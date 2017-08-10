import numpy as np
import pickle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('file', type=str)
parser.add_argument('out_file', type=str)
args = parser.parse_args()

with open(args.file, 'rb') as f:
    params = pickle.load(f)

out = np.array([param.get_value() for param in params])
np.save(args.out_file, out)
