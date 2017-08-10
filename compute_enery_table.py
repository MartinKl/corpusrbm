import numpy as np
from rnnrbm import RnnRbm
import pickle
from theano import tensor as T
from functools import reduce


def params():
    yield from np.load('latest_model.npy')

network = RnnRbm()
prm = params()
for param in network.params:
    param.set_value(next(prm))

with open('cvoc.pkl', 'rb') as f:
    d = {v: k for k, v in pickle.load(f).items()}

W, bv, bh, _, _, _, _, _ = tuple(param.get_value() for param in network.params)


def free_energy(v):
    return -(v * bv).sum() - np.log(1 + np.exp(np.dot(v, W) + bh)).sum()

new_keys = \
{'<',
 '>',
 '\\',
 '{',
 '|',
 '}',
 '~',
 '°',
 'Â',
 'Ê',
 'Î',
 'Ô',
 'Û',
 '÷',
 'ù',
 'Ę',
 'ō',
 'ŷ',
 '́',
 'Α',
 'Γ',
 'Ε',
 'Ι',
 'Λ',
 'Ν',
 'Ο',
 'Τ',
 'Φ',
 'Χ',
 'θ',
 'ο',
 'π'}

eye = np.eye(129, dtype=np.uint8)
energy_dict = {i: free_energy(eye[i]) for i in range(129)}
max_e = max(energy_dict.values())
energy_dict = {ix: e_val / max_e for ix, e_val in energy_dict.items()}
energy = {}
for data_name, f_name in (
        ('0.1', 'c_x.npy'),
        ('notker', 'c_x_complete.npy')
):
    arr = np.load(f_name)
    energy[data_name] = []
    for seq in arr:
        energy[data_name].append(reduce(lambda a, b: a*b, [energy_dict[ix] if ix in d else energy_dict[128] for ix in seq]))

for data_name, values in energy.items():
    print(data_name, end=':\n')
    print(np.mean(values), np.std(values))
    print(np.min(values), np.max(values), end='\n'*2)

