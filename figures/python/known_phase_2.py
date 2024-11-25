"""Generate figure showing standard deviation and worst-case entropy for a single
bit, given the knowledge of n (0 - 10) previous phase values."""

import sys
import argparse
from os import getcwd
from typing import List, cast
import numpy as np
import scipy.stats # type: ignore
sys.path.append(getcwd())
from lib import graph_maker as g_m
from lib import store_data as s_d
from math_model.python import noises

F_N = 520e6
H_W = 18.9e-15
H_F = 9.48e-9
T_ACCS = [10.86e-9, 43.44e-9, 173.76e-9]
MAX_HIST = 5
S_U = 6
T_ACC_MARKERS = ['circle', 'tri_up', 'cross']

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

data_storage = s_d.StoreData(name='known_phase_2')

if args.d:
    w_noise = noises.WhiteFMNoise(F_N, H_W)
    f_noise = noises.FlickerFMNoise(F_N, H_F)

    std_ws: List[List[float]] = []
    std_fs: List[List[float]] = []
    std_ss: List[List[float]] = []
    hw_ws: List[List[float]] = []
    hw_fs: List[List[float]] = []
    hw_ss: List[List[float]] = []
    for t_acc in T_ACCS:
        t_u = t_acc * S_U # Unknown sampling time # pylint: disable=invalid-name
        sigma_uuw = w_noise.kernel([t_u])
        sigma_uuf = f_noise.kernel([t_u])
        nom_p_U = 2 * np.pi * F_N * t_u # Nominal phase at t_u
        std_ws_temp: List[float] = []
        std_fs_temp: List[float] = []
        std_ss_temp: List[float] = []
        hw_ws_temp: List[float] = []
        hw_fs_temp: List[float] = []
        hw_ss_temp: List[float] = []
        for hist_nb in range(MAX_HIST + 1):
            t_os = [t_u - (i + 1) * t_acc for i in range(hist_nb)]
            sigma_uow = w_noise.kernel([t_u], t_os)
            sigma_oow = w_noise.kernel(t_os)
            sigma_ouw = cast(List[List[float]], sigma_uow.T) # type: ignore
            sigma_uof = f_noise.kernel([t_u], t_os)
            sigma_oof = f_noise.kernel(t_os)
            sigma_ouf = cast(List[List[float]], sigma_uof.T) # type: ignore
            sigma_tempw = np.linalg.solve(sigma_oow, sigma_ouw).T
            sigma_tempf = np.linalg.solve(sigma_oof, sigma_ouf).T
            sigma_ucow = cast(float, (sigma_uuw - sigma_tempw @ sigma_ouw)[0][0])
            sigma_ucof = cast(float, (sigma_uuf - sigma_tempf @ sigma_ouf)[0][0])
            sigma_ucos = sigma_ucow + sigma_ucof
            std_ws_temp.append(np.sqrt(sigma_ucow))
            std_fs_temp.append(np.sqrt(sigma_ucof))
            std_ss_temp.append(np.sqrt(sigma_ucos))
            hw_ws_temp.append(s2h(std_ws_temp[-1]))
            hw_fs_temp.append(s2h(std_fs_temp[-1]))
            hw_ss_temp.append(s2h(std_ss_temp[-1]))
        std_ws.append(std_ws_temp)
        std_fs.append(std_fs_temp)
        std_ss.append(std_ss_temp)
        hw_ws.append(hw_ws_temp)
        hw_fs.append(hw_fs_temp)
        hw_ss.append(hw_ss_temp)
    data_to_store: List[List[float]] = std_ws + std_fs + std_ss + hw_ws + hw_fs + hw_ss
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
    std_ws = data_from_store[:len(T_ACCS)]
    std_fs = data_from_store[len(T_ACCS):2 * len(T_ACCS)]
    std_ss = data_from_store[2 * len(T_ACCS):3 * len(T_ACCS)]
    hw_ws = data_from_store[3 * len(T_ACCS):4 * len(T_ACCS)]
    hw_fs = data_from_store[4 * len(T_ACCS):5 * len(T_ACCS)]
    hw_ss = data_from_store[5 * len(T_ACCS):]

graph_maker = g_m.GraphMaker('known_phase_2.svg', folder_name='figures')
graph_maker.create_grid((2, 1), marg_mid_ver=0, marg_right=0.65, marg_left=0.13)
ax_std = graph_maker.create_ax(x_slice=0, y_slice=1, # pylint: disable=invalid-name
                               x_label='Number known previous samples ($p$)',
                               y_label=(r"""$\sqrt{\mathbf{Var} \bigl [ """
                                        r"""\Phi^y_e(t_6) \bigr ]}$"""),
                               y_unit='rad', y_scale='log10', show_legend=True,
                               max_nb_y_ticks=10, max_nb_x_ticks=6,
                               legend_bbox=[1, 0.75, 0.5, 0.5], legend_loc='center left')
ax_h = graph_maker.create_ax(x_slice=0, y_slice=0, # pylint: disable=invalid-name
                             title=(r"""Known previous sample phases, """
                                    r"""$h_f = \num{""" f'{H_F:8.2e}' r"""}$"""),
                             title_loc='left',
                             y_label=r"""$\mathbf{H_{worst}}$""",
                             y_unit='bit', show_legend=False,
                             show_x_labels=False,
                             y_scale='ent')

for index, (t_acc, marker, std_w, std_f, std_s, h_w, h_f, h_s) \
    in enumerate(zip(T_ACCS, T_ACC_MARKERS, std_ws, std_fs, std_ss, hw_ws, hw_fs, hw_ss)):
    graph_maker.plot(ax_h, list(range(MAX_HIST + 1)), h_w, line_style='solid',
                     color=index, marker=marker)
    graph_maker.plot(ax_h, list(range(MAX_HIST + 1)), h_f, line_style='dashed',
                     color=index, marker=marker)
    graph_maker.plot(ax_std, list(range(MAX_HIST + 1)), std_w,
                     line_style='solid', color=index, marker=marker)
    graph_maker.plot(ax_std, list(range(MAX_HIST + 1)), std_f,
                     line_style='dashed', color=index, marker=marker)

# Legend entries:
graph_maker.plot(ax_std, [], [], line_style='solid', color='grey',
                 label='White', visible=False)
graph_maker.plot(ax_std, [], [], line_style='dashed', color='grey',
                 label='Flicker', visible=False)
for index, (t_acc, marker) in enumerate(zip(T_ACCS, T_ACC_MARKERS)):
    graph_maker.plot(ax_std, [], [], line_style='none', color=index,
                     marker=marker, visible=False,
                     label=(r"""$t_{acc} = \num{""" f'{t_acc*1e9:.0f}'
                            r"""} \; n s$"""))

graph_maker.write_svg()
