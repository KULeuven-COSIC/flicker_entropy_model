"""Generate a figure showing the worst-case Shannon entropy, when up to b previous
sample bits are known."""

import sys
import argparse
from os import getcwd
from os.path import join
from typing import List, Dict, Tuple
import numpy as np
sys.path.append(getcwd())
from lib import graph_maker as g_m
from lib import store_data as s_d
from math_model.python import data_reader as d_r

F_N = 520e6
H_W = 18.9e-15
H_F = 6.215e-12
BIT_NB = 300
T_ACCS = [25e-6, 100.015e-6, 400.06e-6]
HIST_DEPTH = 10
T_ACC_MARKERS = ['circle', 'tri_up', 'cross']

def hm2h(hm: float) -> float:
    """Min entropy to Shannon entropy."""
    p_1 = 2**(-hm)
    if (p_1 == 0) | (p_1 == 1):
        return 0
    return -p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1)

parser = argparse.ArgumentParser()
parser.add_argument('-v', help='Print bit probability', action='store_true')
parser.add_argument('-d', help='Run data generation', action='store_true')
parser.add_argument('-q', help='Quit after data collect', action='store_true')
args = parser.parse_args()

data_storage = s_d.StoreData(name='comb_bit_entropy_0')

if args.d:
    dist_man = d_r.DistributionManager(join('math_model', 'simulation_data'))
    h_accs_w: List[List[float]] = []
    h_accs_f: List[List[float]] = []
    for t_acc in T_ACCS:
        dist_datas_dict: Dict[int, Dict[Tuple[int, int], List[Tuple[str, int]]]] = {}
        for depth in range(HIST_DEPTH, -1, -1):
            dist_datas_dict[depth] = dist_man.get_all_hist_depth(F_N, H_F, t_acc,
                                                                 'comb', BIT_NB, depth)
        # hs_r_w: List[float] = [0] * (HIST_DEPTH + 1)
        hs_w_w: List[float] = [0] * (HIST_DEPTH + 1)
        # hs_r_f: List[float] = [0] * (HIST_DEPTH + 1)
        hs_w_f: List[float] = [0] * (HIST_DEPTH + 1)
        ns_w: List[int] = [0] * (HIST_DEPTH + 1)
        ns_f: List[int] = [0] * (HIST_DEPTH + 1)
        for depth in range(HIST_DEPTH, -1, -1):
            if args.v:
                print(depth)
            depth_mult = 2**(HIST_DEPTH - depth)
            inds_add: List[int] = [i * 2**depth for i in range(depth_mult)]
            for l in dist_datas_dict[depth].values():
                for id_, nb_bin in l:
                    _, __, hist_value, white = d_r.DistributionManager.parse_dist_id(id_)
                    dist = dist_man.get_dist(id_, nb_bin, F_N, H_F, t_acc, 'comb')
                    h_r = dist.min_entropy
                    h_w, _ = dist.worst_min_entropy # type: ignore
                    if (h_r < 0) | (h_w < 0):
                        print(f'Negative entropy, value: {hist_value}, '
                                f'depth: {depth}, white: {white}')
                        continue
                    hs_r = hm2h(h_r)
                    hs_w = hm2h(h_w)
                    if white:
                        # hs_r_w[depth] += hs_r * dist.nb_samples
                        hs_w_w[depth] += hs_w * dist.nb_samples
                        ns_w[depth] += dist.nb_samples
                    else:
                        # hs_r_f[depth] += hs_r * dist.nb_samples
                        hs_w_f[depth] += hs_w * dist.nb_samples
                        ns_f[depth] += dist.nb_samples
        # hs_r_w = [hi / ni for hi, ni in zip(hs_r_w, ns_w)]
        hs_w_w = [hi / ni for hi, ni in zip(hs_w_w, ns_w)]
        # hs_r_f = [hi / ni for hi, ni in zip(hs_r_f, ns_f)]
        hs_w_f = [hi / ni for hi, ni in zip(hs_w_f, ns_f)]
        h_accs_w.append(hs_w_w)
        h_accs_f.append(hs_w_f)
    data_to_store: List[List[float]] = h_accs_w + h_accs_f
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
    h_accs_w = data_from_store[:len(T_ACCS)]
    h_accs_f = data_from_store[len(T_ACCS):]

graph_maker = g_m.GraphMaker('comb_bit_entropy_0.svg', folder_name='figures')
graph_maker.create_grid(size=(1, 1), marg_left=0.15)
ax = graph_maker.create_ax(x_slice=0, y_slice=0, # pylint: disable=invalid-name
                           title=(r"""Known previous sample bits, $h_f = \num{"""
                                  f'{np.round(H_F * 1e14) / 1e14:.2e}' r"""}$"""), # type: ignore
                           x_label='Number known previous bits ($p$)',
                           y_label='Worst-case Shannon entropy',
                           y_unit='bit',
                           y_scale='ent',
                           show_legend=True,
                           legend_loc='lower left')

for index, (t_acc, marker, h_fs, h_ws) \
    in enumerate(zip(T_ACCS, T_ACC_MARKERS, h_accs_f, h_accs_w)):
    graph_maker.plot(ax=ax, xs=list(range(HIST_DEPTH + 1)), ys=h_ws,
                     color=index, marker=marker,
                     line_style='solid')
    graph_maker.plot(ax=ax, xs=list(range(HIST_DEPTH + 1)), ys=h_fs,
                     color=index, marker=marker,
                     line_style='dashed')
# Legend entries:
graph_maker.plot(ax=ax, xs=[], ys=[], line_style='solid',
                 color='grey', label='White', visible=False)
graph_maker.plot(ax=ax, xs=[], ys=[], line_style='dashed',
                 color='grey', label='Flicker', visible=False)
for index, (t_acc, marker) in enumerate(zip(T_ACCS, T_ACC_MARKERS)):
    graph_maker.plot(ax=ax, xs=[], ys=[], line_style='none',
                     color=index, marker=marker,
                     label=(r"""$t_{acc} = \num{"""
                            f'{int(np.round(t_acc * 1e6)):d}' r"""} \; \mu s$"""), # type: ignore
                     visible=False)

graph_maker.write_svg()
