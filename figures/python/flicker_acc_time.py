"""Generate a figure showing the worst-case entropy, given the knowledge of the previous
phase for flicker noise versus the accumulation time between the bits. Show curves for
multiple flicker noise magnitudes."""

import sys
import argparse
from os import getcwd
from typing import List, cast, Dict, Any
import numpy as np
import scipy.stats # type: ignore
sys.path.append(getcwd())
from lib import graph_maker as g_m
from lib import store_data as s_d
from math_model.python import noises

F_N = 520e6
H_W = 18.9e-15
H_FS = [9.48e-9, 103.85e-12, 6.216e-12]
T_ACC_MIN = 1e-7
T_ACC_MAX = 1e-4
NB_POINTS = 1000
BIT_NBS = [6, 1e3, 1e6]
MAIN_BIT_NB = 6
BIT_NB_LINE_STYLES = ['solid', 'dashed', 'dashdotted', 'dashdotdotted']

t_accs = cast(List[float], np.logspace(np.log10(T_ACC_MIN), np.log10(T_ACC_MAX),
                                       NB_POINTS, endpoint=True))

def s2h(std: float) -> float:
    """Convert std to worst-case H."""
    rv = scipy.stats.norm(0, std)
    p_1 = cast(float, rv.cdf(np.pi / 2) - 0.5) # type: ignore
    pi_start = np.pi * 3 / 2
    while pi_start < 5 * std:
        p_1 += cast(float, (rv.cdf(pi_start + np.pi) - rv.cdf(pi_start))) # type: ignore
        pi_start += 2 * np.pi
    p_1 *= 2
    if (p_1 == 0) | (p_1 == 1):
        return 0
    return -p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1)

parser = argparse.ArgumentParser()
parser.add_argument('-v', help='Print bit probability', action='store_true')
parser.add_argument('-d', help='Run data generation', action='store_true')
parser.add_argument('-q', help='Quit after data collect', action='store_true')
args = parser.parse_args()

data_storage = s_d.StoreData(name='flicker_acc_time')

if args.d:
    w_noise = noises.WhiteFMNoise(F_N, H_W)
    h_w_bs: List[List[float]] = []
    h_f_bss: List[List[List[float]]] = []
    for bit_nb in BIT_NBS:
        h_ws: List[float] = []
        h_fss: List[List[float]] = [[] for _ in H_FS]
        for t_acc in t_accs:
            t_u = bit_nb * t_acc
            t_o = t_u - t_acc
            var_ucow = w_noise.auto_cor(t_u, t_u) \
                - w_noise.auto_cor(t_u, t_o)**2 / w_noise.auto_cor(t_o, t_o)
            h_ws.append(s2h(np.sqrt(var_ucow)))
            for index, h_f in enumerate(H_FS):
                f_noise = noises.FlickerFMNoise(F_N, h_f)
                var_ucof = f_noise.auto_cor(t_u, t_u) \
                    - f_noise.auto_cor(t_u, t_o)**2 / f_noise.auto_cor(t_o, t_o)
                h_fss[index].append(s2h(np.sqrt(var_ucof)))
        h_w_bs.append(h_ws)
        h_f_bss.append(h_fss)
    data_to_store: List[List[float]] = h_w_bs
    for h_fs in h_f_bss:
        data_to_store += h_fs
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
    h_w_bs = data_from_store[:len(BIT_NBS)]
    h_f_bss = []
    for index, _ in enumerate(BIT_NBS):
        h_f_bss.append(data_from_store[len(BIT_NBS) + index * len(H_FS):
                                       len(BIT_NBS) + (index + 1) * len(H_FS)])

graph_maker = g_m.GraphMaker('flicker_acc_time.svg', folder_name='figures')
graph_maker.create_grid((1, 1), marg_left=0.13)
ax = graph_maker.create_ax(x_slice=0, y_slice=0, # pylint: disable=invalid-name
                           title='Worst-case entropy versus accumulation time',
                           x_label=r"""Accumulation time ($t_{acc}$)""",
                           y_label=r"""Worst-case Shannon entropy""",
                           x_unit='s', y_unit='bit',
                           x_scale='log10',
                           y_scale='ent',
                           show_legend=True,
                           legend_loc='upper left')

graph_maker.plot(ax=ax, xs=t_accs, ys=h_w_bs[0], line_style='solid',
                 label='White', color='dark_grey', zorder=1) # All bit nbs give the same white noise
for b_line_style, bit_nb, h_fss in zip(BIT_NB_LINE_STYLES, BIT_NBS, h_f_bss):
    for index, (h_f, h_fs_i) in enumerate(zip(H_FS, h_fss)):
        plt_kwargs: Dict[str, Any] = {
            'ax': ax,
            'xs': t_accs,
            'ys': h_fs_i,
            'color': index
        }
        if bit_nb == MAIN_BIT_NB:
            plt_kwargs['line_style'] = 'solid'
            plt_kwargs['zorder'] = 2
        else:
            plt_kwargs['line_style'] = b_line_style
            plt_kwargs['alpha'] = 0.7
            plt_kwargs['zorder'] = 0
        graph_maker.plot(**plt_kwargs)

# Legend entries:
for b_line_style, bit_nb in zip(BIT_NB_LINE_STYLES, BIT_NBS):
    if bit_nb >= 100:
        bit_nb_str = r"""10\textsuperscript{""" f'{int(np.log10(bit_nb)):d}' r"""}"""
    else:
        bit_nb_str = f'{bit_nb:d}'
    plt_kwargs = {
        'ax': ax,
        'xs': [],
        'ys': [],
        'color': 'grey',
        'label': f'Flicker, {bit_nb_str}',
        'visible': False
    }
    if bit_nb == MAIN_BIT_NB:
        plt_kwargs['line_style'] = 'solid'
    else:
        plt_kwargs['line_style'] = b_line_style
        plt_kwargs['alpha'] = 0.7
    graph_maker.plot(**plt_kwargs)

graph_maker.write_svg()
