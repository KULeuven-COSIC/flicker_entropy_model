"""Generate a figure showing the Gaussian process in action for
both white FM and flicker FM noise."""

import sys
import argparse
from os import getcwd
from typing import List, cast
import numpy as np
sys.path.append(getcwd())
from lib import graph_maker as g_m # pylint: disable=wrong-import-position
from lib import store_data as s_d # pylint: disable=wrong-import-position
from math_model.python import noises

F_N = 520e6
H_W = 18.9e-15
H_F = 1e-10
T_ACC_MIN = 0
T_ACC_MAX = 10e-6
NB_POINTS = 100
FIXED_POINTS = [1e-6, 3e-6, 4e-6]
STD_MULT = 1

t_accs = cast(List[float], np.linspace(T_ACC_MIN, T_ACC_MAX,
                                       NB_POINTS, endpoint=True))
for f_point in FIXED_POINTS:
    if f_point in t_accs:
        index = np.where(t_accs == f_point)[0][0] # type: ignore
        t_accs = np.delete(t_accs, index) # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument('-v', help='Print bit probability', action='store_true')
parser.add_argument('-d', help='Run data generation', action='store_true')
parser.add_argument('-q', help='Quit after data collect', action='store_true')
args = parser.parse_args()

data_storage = s_d.StoreData(name='gauss_process')

if args.d:
    w_noise = noises.WhiteFMNoise(F_N, H_W)
    f_noise = noises.FlickerFMNoise(F_N, H_F)
    w_gen = noises.NoiseGenInst(w_noise, False)
    f_gen = noises.NoiseGenInst(f_noise, False)
    w_gen.add_time_points(FIXED_POINTS)
    f_gen.add_time_points(FIXED_POINTS)
    w_m, w_s = w_gen.get_cond(t_accs, update_sigma=False) # type: ignore
    f_m, f_s = f_gen.get_cond(t_accs, update_sigma=False) # type: ignore
    w_m, f_m = [m for m in w_m], [m for m in f_m]
    w_std = np.sqrt(np.diag(w_s))
    f_std = np.sqrt(np.diag(f_s))
    w_gen_time, f_gen_time = w_gen.time, f_gen.time
    w_gen_phase, f_gen_phase = w_gen.phase, f_gen.phase
    data_to_store: List[List[float]] = [w_m, f_m, w_std, f_std,
                                        w_gen_time, f_gen_time, w_gen_phase, f_gen_phase]
    data_storage.write_data(data_to_store, over_write=True)
    if args.q:
        sys.exit()
else:
    if not data_storage.file_exist:
        print(f'File: {data_storage.file_path} does not exist. '
              f'Run with \"d\" option to generate data.')
        sys.exit()
    data_from_store = data_storage.read_data()
    assert data_from_store is not None
    w_m, f_m, w_std, f_std, \
    w_gen_time, f_gen_time, w_gen_phase, f_gen_phase = data_from_store

graph_maker = g_m.GraphMaker('gauss_process.svg', folder_name='figures')
graph_maker.create_grid(size=(1, 1), marg_left=0.1)
ax = graph_maker.create_ax(x_slice=0, y_slice=0, # pylint: disable=invalid-name
                           title='White FM and flicker FM noise Gaussian process',
                           x_label='Time ($t$)',
                           y_label=r"""Excess phase ($\Phi^y_e(t)$)""",
                           x_unit='s', y_unit='rad',
                           show_legend=True,
                           legend_loc='upper left')

graph_maker.fill_between_y(ax=ax, xs=t_accs,
                           y0s=[mi - STD_MULT * si for mi, si in zip(w_m, w_std)],
                           y1s=[mi + STD_MULT * si for mi, si in zip(w_m, w_std)],
                           color=0)
graph_maker.fill_between_y(ax=ax, xs=t_accs,
                           y0s=[mi - STD_MULT * si for mi, si in zip(f_m, f_std)],
                           y1s=[mi + STD_MULT * si for mi, si in zip(f_m, f_std)],
                           color=1)
graph_maker.plot(ax=ax, xs=t_accs, ys=w_m,
                 color=0)
graph_maker.plot(ax=ax, xs=t_accs, ys=f_m,
                 color=1)
graph_maker.plot(ax=ax, xs=w_gen_time, ys=w_gen_phase,
                 line_style='none',
                 marker_color=0,
                 marker='plus',
                 marker_edge_color='white',
                 line_width=1.5,
                 label=r"""$\varphi^w_e(t_i)$""")
graph_maker.plot(ax=ax, xs=f_gen_time, ys=f_gen_phase,
                 line_style='none',
                 marker_color=1,
                 marker='cross',
                 marker_edge_color='white',
                 line_width=1.5,
                 label=r"""$\varphi^f_e(t_i)$""")

# Legend entries:
if STD_MULT != 1:
    std_mult_str = r"""\num{""" f'{STD_MULT:d}' r"""} \times"""
else:
    std_mult_str = '' # pylint: disable=invalid-name
graph_maker.fill_between_y(ax=ax, xs=[], y0s=[],
                           color='grey',
                           label=(f'${std_mult_str}'
                                  r""" \sqrt{\mathbf{Var} \bigl [ \Phi^y_e(t) \bigr ]}$"""))
graph_maker.plot(ax=ax, xs=[], ys=[],
                 color='grey',
                 label=r"""$\mathbf{E} \bigl [ \Phi^y_e(t) \bigr ]$""")

graph_maker.write_svg()
