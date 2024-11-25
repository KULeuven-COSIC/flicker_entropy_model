"""Generate a figure showing a multivariate Gaussian PDF, for two samples. Highlight how
the PDF changes when the first sample turns out to be equal to one."""

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
T_ACC = 4.11e-6 # 10.11e-6
H_0 = 18.9e-15
H_M1 = 100e-12
S_0 = 1
S_1 = 2
NB_POINTS = 25 # 150 # Figure resolution
SIGMA_MULT = 4

parser = argparse.ArgumentParser()
parser.add_argument('-v', help='Print process', action='store_true')
parser.add_argument('-d', help='Run data generation', action='store_true')
parser.add_argument('-q', help='Quit after data collect', action='store_true')
args = parser.parse_args()

if args.d:
    if args.q:
        sys.exit()

w_noise = noises.WhiteFMNoise(F_N, H_0)
f_noise = noises.FlickerFMNoise(F_N, H_M1)

t0, t1 = T_ACC * S_0, T_ACC * S_1 # Sampling times
nom_p_0 = 2 * np.pi * F_N * t0 # Nominal phase at t0
nom_p_1 = 2 * np.pi * F_N * t1 # Nominal phase at t1

s_w = np.array([[w_noise.auto_cor(t0, t0), w_noise.auto_cor(t0, t1)],
                [w_noise.auto_cor(t1, t0), w_noise.auto_cor(t1, t1)]]) # pylint: disable=arguments-out-of-order
s_f = np.array([[f_noise.auto_cor(t0, t0), f_noise.auto_cor(t0, t1)],
                [f_noise.auto_cor(t1, t0), f_noise.auto_cor(t1, t1)]]) # pylint: disable=arguments-out-of-order
s_s = np.block([[s_w, np.zeros((2, 2))], [np.zeros((2, 2)), s_f]])

norm_dist = scipy.stats.multivariate_normal(mean=np.zeros(4), cov=s_s) # type: ignore

w_0s = cast(List[float], np.linspace(-SIGMA_MULT * s_w[0, 0] * 1.2,
                                     SIGMA_MULT * s_w[0, 0] * 1.2, NB_POINTS, endpoint=True))
w_1s = cast(List[float], np.linspace(-SIGMA_MULT * s_w[1, 1],
                                     SIGMA_MULT * s_w[1, 1], NB_POINTS, endpoint=True))
f_0s = cast(List[float], np.linspace(-SIGMA_MULT * s_f[0, 0] * 1.2,
                                     SIGMA_MULT * s_f[0, 0] * 1.2, NB_POINTS, endpoint=True))
f_1s = cast(List[float], np.linspace(-SIGMA_MULT * s_f[1, 1] * 0.7,
                                     SIGMA_MULT * s_f[1, 1] * 0.7, NB_POINTS, endpoint=True))

d_w0 = w_0s[1] - w_0s[0]
d_w1 = w_1s[1] - w_1s[0]
d_f0 = f_0s[1] - f_0s[0]
d_f1 = f_1s[1] - f_1s[0]

pdf_array = np.zeros((NB_POINTS, NB_POINTS, NB_POINTS, NB_POINTS))
pdf_array_bit_1 = np.zeros((NB_POINTS, NB_POINTS, NB_POINTS, NB_POINTS))
for i_w0, w_0i in enumerate(w_0s):
    if args.v:
        print(i_w0)
    for i_w1, w_1i in enumerate(w_1s):
        for i_f0, f_0i in enumerate(f_0s):
            for i_f1, f_1i in enumerate(f_1s):
                p = norm_dist.pdf(np.array([w_0i, w_1i, f_0i, f_1i])) # type: ignore
                pdf_array[i_w0, i_w1, i_f0, i_f1] = p
                if int(np.floor((w_0i + f_0i + nom_p_0) / np.pi)) % 2 == 1:
                    pdf_array_bit_1[i_w0, i_w1, i_f0, i_f1] = p
prob_1 = sum(sum(sum(sum(pdf_array_bit_1)))) * d_w0 * d_w1 * d_f0 * d_f1 # type: ignore
if args.v:
    print(f'Prob 1 = {prob_1}')
for i_w0 in range(NB_POINTS):
    for i_w1 in range(NB_POINTS):
        for i_f0 in range(NB_POINTS):
            for i_f1 in range(NB_POINTS):
                pdf_array_bit_1[i_w0, i_w1, i_f0, i_f1] /= prob_1 # type: ignore

image_w0_f0 = [[sum((sum((pdf_array[i_w0][i_w1][i_f0][i_f1]
                          for i_f1 in range(NB_POINTS))) * d_f1
                          for i_w1 in range(NB_POINTS))) * d_w1
                          for i_w0 in range(NB_POINTS)] for i_f0 in range(NB_POINTS)]
image_w0_f0_1 = [[sum((sum((pdf_array_bit_1[i_w0][i_w1][i_f0][i_f1]
                            for i_f1 in range(NB_POINTS))) * d_f1
                            for i_w1 in range(NB_POINTS))) * d_w1
                            for i_w0 in range(NB_POINTS)] for i_f0 in range(NB_POINTS)]
image_w1_f1 = [[sum((sum((pdf_array[i_w0][i_w1][i_f0][i_f1]
                          for i_f0 in range(NB_POINTS))) * d_f0
                          for i_w0 in range(NB_POINTS))) * d_w0
                          for i_w1 in range(NB_POINTS)] for i_f1 in range(NB_POINTS)]
image_w1_f1_1 = [[sum((sum((pdf_array_bit_1[i_w0][i_w1][i_f0][i_f1]
                            for i_f0 in range(NB_POINTS))) * d_f0
                            for i_w0 in range(NB_POINTS))) * d_w0
                            for i_w1 in range(NB_POINTS)] for i_f1 in range(NB_POINTS)]

pdf_w0_c1 = [sum((image_w0_f0_1[i_f0][i_w0]
                  for i_f0 in range(NB_POINTS))) * d_f0
                  for i_w0 in range(NB_POINTS)]
pdf_f0_c1 = [sum((image_w0_f0_1[i_f0][i_w0]
                  for i_w0 in range(NB_POINTS))) * d_w0
                  for i_f0 in range(NB_POINTS)]
pdf_w1_c1 = [sum((image_w1_f1_1[i_f1][i_w1]
                  for i_f1 in range(NB_POINTS))) * d_f1
                  for i_w1 in range(NB_POINTS)]
pdf_f1_c1 = [sum((image_w1_f1_1[i_f1][i_w1]
                  for i_w1 in range(NB_POINTS))) * d_w1
                  for i_f1 in range(NB_POINTS)]

pdf_w0 = [sum((image_w0_f0[i_f0][i_w0]
                  for i_f0 in range(NB_POINTS))) * d_f0
                  for i_w0 in range(NB_POINTS)]
pdf_f0 = [sum((image_w0_f0[i_f0][i_w0]
                  for i_w0 in range(NB_POINTS))) * d_w0
                  for i_f0 in range(NB_POINTS)]
pdf_w1 = [sum((image_w1_f1[i_f1][i_w1]
                  for i_f1 in range(NB_POINTS))) * d_f1
                  for i_w1 in range(NB_POINTS)]
pdf_f1 = [sum((image_w1_f1[i_f1][i_w1]
                  for i_w1 in range(NB_POINTS))) * d_w1
                  for i_f1 in range(NB_POINTS)]

if args.v:
    print('PDF sanity check:')
    print(sum(pdf_w0_c1) * d_w0, sum(pdf_f0_c1) * d_f0,
          sum(pdf_w1_c1) * d_w1, sum(pdf_f1_c1) * d_f1)
    print(sum(pdf_w0) * d_w0, sum(pdf_f0) * d_f0,
          sum(pdf_w1) * d_w1, sum(pdf_f1) * d_f1)

diff_w0_f0 = [[p_1 - p for p_1, p in zip(row_1, row)]
              for row_1, row in zip(image_w0_f0_1, image_w0_f0)]
diff_w1_f1 = [[p_1 - p for p_1, p in zip(row_1, row)]
              for row_1, row in zip(image_w1_f1_1, image_w1_f1)]

pdf_diff_w0 = [p_1 - p for p_1, p in zip(pdf_w0_c1, pdf_w0)]
pdf_diff_f0 = [p_1 - p for p_1, p in zip(pdf_f0_c1, pdf_f0)]
pdf_diff_w1 = [p_1 - p for p_1, p in zip(pdf_w1_c1, pdf_w1)]
pdf_diff_f1 = [p_1 - p for p_1, p in zip(pdf_f1_c1, pdf_f1)]

# #clip image:
# image_w0_f0 = [[max(p, 1e-150) for p in row] for row in image_w0_f0]
# image_w1_f1 = [[max(p, 1e-150) for p in row] for row in image_w1_f1]

graph_maker = g_m.GraphMaker('two_diff.svg', figure_size=(3, 3), folder_name='figures')
graph_maker.create_grid(size=(3, 11), x_ratios=[1, 0.1, 3, 0.1, 0.5, 2.5, 1, 0.1, 3, 0.1, 0.5],
                        y_ratios=[3, 0.1, 0.6], marg_mid_hor=0, marg_mid_ver=0,
                        marg_bot=0.13, marg_left=0.12, marg_right=0.93, marg_top=0.88)

ax_w0_f0 = graph_maker.create_ax(2, 0, show_x_labels=False, show_y_labels=False, # pylint: disable=invalid-name
                                 title=('PDF difference at $t_1$'), title_pad=10)
ax_w1_f1 = graph_maker.create_ax(8, 0, show_x_labels=False, show_y_labels=False, # pylint: disable=invalid-name
                                 title=('PDF difference at $t_2$'), title_pad=10)

ax_c_bar_0 = graph_maker.create_ax(4, 0) # pylint: disable=invalid-name
ax_c_bar_1 = graph_maker.create_ax(10, 0) # pylint: disable=invalid-name

ax_w0 = graph_maker.create_ax(2, 2, show_y_ticks=False, share_x=ax_w0_f0, # pylint: disable=invalid-name
                              x_scale='pi', max_nb_x_ticks=6,
                              x_label='White phase ($\\varphi^w$)')
ax_f0 = graph_maker.create_ax(0, 0, show_x_ticks=False, share_y=ax_w0_f0, # pylint: disable=invalid-name
                              y_scale='pi', max_nb_y_ticks=8,
                              y_label='Flicker phase ($\\varphi^f$)')
ax_w1 = graph_maker.create_ax(8, 2, show_y_ticks=False, share_x=ax_w1_f1, # pylint: disable=invalid-name
                              x_scale='pi', max_nb_x_ticks=6,
                              x_label='White phase ($\\varphi^w$)')
ax_f1 = graph_maker.create_ax(6, 0, show_x_ticks=False, share_y=ax_w1_f1, # pylint: disable=invalid-name
                              y_scale='pi', max_nb_y_ticks=8,
                              y_label='Flicker phase ($\\varphi^f$)')

graph_maker.contour_fill(ax_w0_f0, diff_w0_f0, w_0s, f_0s,
                         color_norm=g_m.GraphMaker.ColorNorm.SYMLOG,
                         color_bar=ax_c_bar_0,
                        #  norm_vmin=-1, norm_vmax=1,
                         color_map=g_m.GraphMaker.ColorMap.BLUE_WHITE_RED)
graph_maker.contour_fill(ax_w1_f1, diff_w1_f1, w_1s, f_1s,
                         color_norm=g_m.GraphMaker.ColorNorm.SYMLOG,
                         color_bar=ax_c_bar_1,
                        #  norm_vmin=-1, norm_vmax=1,
                         color_map=g_m.GraphMaker.ColorMap.BLUE_WHITE_RED)

graph_maker.plot(ax_w0, w_0s, pdf_diff_w0)
graph_maker.plot(ax_f0, pdf_diff_f0, f_0s)
graph_maker.plot(ax_w1, w_1s, pdf_diff_w1)
graph_maker.plot(ax_f1, pdf_diff_f1, f_1s)

graph_maker.write_svg()
