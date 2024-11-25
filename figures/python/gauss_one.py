"""Generate a figure showing a Gaussian PDF, with an integration area under the
curve to visualize the probability of sampling a one."""

import sys
import argparse
from os import getcwd
from typing import List, cast
import numpy as np
import scipy.stats # type: ignore
sys.path.append(getcwd())
from lib import graph_maker as g_m
from math_model.python import noises

F_N = 520e6
T_ACC = 4.11e-6
H_0 = 18.9e-15
H_M1 = 100e-12

parser = argparse.ArgumentParser()
parser.add_argument('-v', help='Print bit probability', action='store_true')
parser.add_argument('-d', help='Run data generation', action='store_true')
parser.add_argument('-q', help='Quit after data collect', action='store_true')
args = parser.parse_args()

if args.d:
    if args.q:
        sys.exit()

w_noise = noises.WhiteFMNoise(F_N, H_0)
f_noise = noises.FlickerFMNoise(F_N, H_M1)

w_var_1 = w_noise.auto_cor(T_ACC, T_ACC)
f_var_1 = f_noise.auto_cor(T_ACC, T_ACC)
s_var_1 = w_var_1 + f_var_1
nom_p_1 = 2 * np.pi * F_N * T_ACC
s_sig_1 = np.sqrt(s_var_1)

norm_dist = scipy.stats.norm(loc=nom_p_1, scale=s_sig_1)

if args.v:
    p_1 = 0 # pylint: disable=invalid-name
    for n in range(int(np.floor((nom_p_1 - 5 * s_sig_1 - np.pi) / 2 / np.pi)),
                int(np.ceil((nom_p_1 + 5 * s_sig_1 - 2 * np.pi) / 2 / np.pi)) + 1):
        p_1 += norm_dist.cdf((2 * n + 2) * np.pi) # type: ignore
        p_1 -= norm_dist.cdf((2 * n + 1) * np.pi) # type: ignore
    print(f'Probability bit equals 1: {p_1*100:.3f} %.')

xs = cast(List[float], list(np.linspace(nom_p_1 - 5 * s_sig_1, # type: ignore
                                        nom_p_1 + 5 * s_sig_1, 1000).flat))
ys = cast(List[float], [norm_dist.pdf(xi) for xi in xs]) # type: ignore
fill_where = [int(xi / np.pi) % 2 == 1 for xi in xs]

graph_maker = g_m.GraphMaker('gauss_one.svg', folder_name='figures')
graph_maker.create_grid()
ax = graph_maker.create_ax(title='Oscillator phase PDF', # pylint: disable=invalid-name
                           x_label=r'Oscillator phase ($\varphi$)',
                           y_label='Probability density', x_unit='rad',
                           y_unit=r'rad\textsuperscript{-1}',
                           x_scale='pi',
                           show_legend=True)
graph_maker.plot(ax, xs, ys, label=r'$f_{\Phi(t_1)}(\varphi)$')
graph_maker.fill_between_y(ax, xs, ys, where=fill_where, label=r"""$B'_1 = 1$""")
graph_maker.write_svg()
