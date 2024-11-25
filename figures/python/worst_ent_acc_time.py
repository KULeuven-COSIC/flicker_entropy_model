"""Generate a figure showing the relation worst entropy vs. accumulation
time. For different white noise strengths."""

import sys
import argparse
from os import getcwd
from typing import List, cast, Dict, Any
import numpy as np
import scipy.stats # type: ignore
sys.path.append(getcwd())
from lib import graph_maker as g_m
from math_model.python import noises

F_N = 520e6
H_0S = [5e-15, 10e-15, 40e-15, 80e-15, 160e-15, 18.9e-15]
H_0_SEL = 5
T_ACC_MIN = 1e-7
T_ACC_MAX = 1e-4
NB_POINTS = 100

parser = argparse.ArgumentParser()
parser.add_argument('-v', help='Print bit probability', action='store_true')
parser.add_argument('-d', help='Run data generation', action='store_true')
parser.add_argument('-q', help='Quit after data collect', action='store_true')
args = parser.parse_args()

if args.d:
    if args.q:
        sys.exit()

t_accs = cast(List[float], np.logspace(np.log10(T_ACC_MIN), np.log10(T_ACC_MAX), # type: ignore
                                       NB_POINTS, endpoint=True))
entropys: List[List[float]] = []
for h_0 in H_0S:
    if args.v:
        print(h_0)
    w_noise = noises.WhiteFMNoise(F_N, h_0)
    ents: List[float] = []
    for t_i in t_accs:
        v = w_noise.auto_cor(t_i, t_i)
        std = np.sqrt(v)
        rv = scipy.stats.norm(0, std)
        p_1 = cast(float, rv.cdf(np.pi / 2) - 0.5) # type: ignore
        int_start = 3 * np.pi / 2
        while int_start < 5 * std:
            p_1 += cast(float, (rv.cdf(int_start + np.pi) - rv.cdf(int_start))) # type: ignore
            int_start += 2 * np.pi
        p_1 *= 2
        if (p_1 == 0) | (p_1 == 1):
            ents.append(0)
        else:
            ents.append(-p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1))
    entropys.append(ents)

graph_maker = g_m.GraphMaker('worst_ent_acc_time.svg', folder_name='figures')
graph_maker.create_grid(marg_left=0.13)
ax = graph_maker.create_ax(title='Worst-case entropy versus accumulation time', # pylint: disable=invalid-name
                           x_label=r'Accumulation time ($t_{acc}$)',
                           y_label='Worst-case Shannon entropy', x_unit='s',
                           y_unit=r'bit',
                           x_scale='log10',
                           y_scale='ent',
                           show_legend=True)
first_kwargs: Dict[str, Any] = {}
for index, (h_0, ents) in enumerate(zip(H_0S, entropys)):
    kwargs: Dict[str, Any] = {
        'ax': ax,
        'xs': t_accs,
        'ys': ents
    }
    if index == H_0_SEL:
        kwargs['color'] = 1
        kwargs['label'] = (r"""$h_w = \SI{""" f'{h_0 * 1e15:4.1f}'
                           r"""}{\femto \second}$""")
    else:
        kwargs['alpha'] = 0.5
        kwargs['line_style'] = 'dashed'
        kwargs['color'] = 'grey'
    if not index:
        first_kwargs = kwargs
    graph_maker.plot(**kwargs)
first_kwargs['label'] = r"""Other $h_w$"""
first_kwargs['visible'] = False
graph_maker.plot(**first_kwargs)

graph_maker.write_svg()
