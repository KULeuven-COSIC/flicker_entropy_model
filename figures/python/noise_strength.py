"""Generate a noise strength plot, comparing the three different flicker noise
strengths used and showing the three different white-flicker noise corners."""

import sys
import argparse
from os import getcwd
from os.path import join
from typing import List, cast, Tuple, Type
import numpy as np
sys.path.append(getcwd())
from lib import graph_maker as g_m
from math_model.python import noises
from math_model.python import jitter_gen as j_g

FN = 520e6
FL = 1e-3
FH = 10e9
H_WHITE = 1.89e-14
H_FLICKERS = [9.48e-9, 103.85e-12, 6.215e-12]
NOISE_CLASSES: List[Type[noises.Noise]] = [noises.WhiteFMNoise, noises.FlickerFMNoise]

T_START = 1e-12
T_END = 1
NB_ACCS = 1000
NB_EXP = 1000

parser = argparse.ArgumentParser()
parser.add_argument('-v', help='Print bit probability', action='store_true')
parser.add_argument('-d', help='Run data generation', action='store_true')
parser.add_argument('-q', help='Quit after data collect', action='store_true')
args = parser.parse_args()

if args.d:
    if args.q:
        sys.exit()

acc_ts = cast(List[float], np.logspace(np.log10(T_START), np.log10(T_END), NB_ACCS))
freq_bound = noises.Noise.FreqBound(FL, FH)

phase_vars: List[List[float]] = []
theo_vars: List[List[List[float]]] = []
noise_corners: List[List[Tuple[str, float, float]]] = []
for i_h_f, h_f in enumerate(H_FLICKERS):
    gen_data = noises.GenerateData(FN, freq_bound, [H_WHITE, h_f],
                                   NOISE_CLASSES, False)
    jit_gen = j_g.JitterGen(gen_data, acc_ts, db_folder=join('math_model', 'simulation_data', 'jitter_gen'),
                            verbose=args.v)
    j_data = jit_gen.jit_man.get_data()
    if args.v:
        print(f'h_f = {h_f}, found {len(j_data)} experiments.')
    phase_vars_i: List[float] = [0] * NB_ACCS
    for phase_index in range(NB_ACCS):
        phase_vars_i[phase_index] = np.var([exp[phase_index] # type: ignore
                                            for exp in j_data])
    phase_vars.append(phase_vars_i)
    theo_vars.append(jit_gen.jit_man.calc_theoretical_phase_vars())
    noise_corners.append(jit_gen.jit_man.calc_noise_corners(theo_vars[-1]))

graph_maker = g_m.GraphMaker('noise_strength.svg', folder_name='figures')
graph_maker.create_grid(marg_left=0.11)
ax = graph_maker.create_ax(title='Accumulated phase variance', # pylint: disable=invalid-name
                           x_label=r'Accumulation time ($t_{acc}$)',
                           y_label='Phase variance', x_unit='s',
                           y_unit=r'rad\textsuperscript{2}',
                           x_scale='log10',
                           y_scale='log10',
                           show_legend=True)
for index, (_vars, h_f) in enumerate(zip(phase_vars, H_FLICKERS)):
    graph_maker.plot(ax, acc_ts, _vars,
                    #  label=(r"""$h_{f} = \num{""" f'{h_f:9.3e}' r"""}$"""),
                     alpha=0.5, line_width=2, color=index)

min_var = min(theo_vars[0][0])
for index, theo_var in enumerate(theo_vars):
    start_index = 0 # pylint: disable=invalid-name
    while theo_var[1][start_index] < min_var:
        start_index += 1
    graph_maker.plot(ax, acc_ts[start_index:], theo_var[1][start_index:], line_style='dashed',
                     color=index)
# Simulation legend entry:
graph_maker.plot(ax, acc_ts, theo_vars[0][0], alpha=0.5, line_width=2, color='grey',
                 label='Simulated', visible=False)
# Theoretical white + legend entry:
graph_maker.plot(ax, acc_ts, theo_vars[0][0],
                 line_style='dotted', color='dark_grey',
                 label=r"""$\mathbf{Var} \bigl [ \Phi^w_e(t_{acc}) \bigr ]$""")
# Theoretical flicker legend entry:
graph_maker.plot(ax, acc_ts, theo_vars[0][0], line_style='dashed', color='grey', visible=False,
                 label=r"""$\mathbf{Var} \bigl [ \Phi^f_e(t_{acc}) \bigr ]$""")
# Noise corner legend entry:
graph_maker.plot(ax, [noise_corners[0][0][1]], [noise_corners[0][0][2]],
                 marker='circle', line_width=1.5, visible=False, label='Noise corner',
                 line_style='none', marker_color='grey', marker_edge_color='white')
for index, noise_corner in enumerate(noise_corners):
    # graph_maker.plot(ax, [noise_corner[0][1]], [noise_corner[0][2]], color='white',
    #                  marker='circle', line_width=1.3)
    graph_maker.plot(ax, [noise_corner[0][1]], [noise_corner[0][2]],
                     marker='circle', line_width=1.5, marker_color=index,
                     marker_edge_color='white')

graph_maker.write_svg()
