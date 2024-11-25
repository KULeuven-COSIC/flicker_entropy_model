"""Generate a figure showing the excess phase PDF for a second sample, when
the first sample bit is known to equal one. Also show an integration area
under the curve for the worst case entropy, to visualize the probability of
sampling a second one."""

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
T_ACC = 4.11e-6
H_0 = 18.9e-15
H_M1 = 100e-12
S_0 = 1
S_1 = 2
NB_POINTS = 25 #150 # Figure resolution
SIGMA_MULT = 4

parser = argparse.ArgumentParser()
parser.add_argument('-v', help='Print bit probability', action='store_true')
parser.add_argument('-d', help='Run data generation', action='store_true')
parser.add_argument('-q', help='Quit after data collect', action='store_true')
args = parser.parse_args()

data_storage = s_d.StoreData(name='sample_two_worst')

if args.d:
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
    prob_1: float = sum(sum(sum(sum(pdf_array_bit_1)))) * d_w0 * d_w1 * d_f0 * d_f1 # type: ignore
    if args.v:
        print(f'Prob 1 = {prob_1}')
    for i_w0 in range(NB_POINTS):
        for i_w1 in range(NB_POINTS):
            for i_f0 in range(NB_POINTS):
                for i_f1 in range(NB_POINTS):
                    pdf_array_bit_1[i_w0, i_w1, i_f0, i_f1] /= prob_1

    image_w1_f1_1 = [[sum((sum((pdf_array_bit_1[i_w0][i_w1][i_f0][i_f1]
                                for i_f0 in range(NB_POINTS))) * d_f0
                                for i_w0 in range(NB_POINTS))) * d_w0
                                for i_w1 in range(NB_POINTS)] for i_f1 in range(NB_POINTS)]

    # pdf_w1_c1 = [sum((image_w1_f1_1[i_f1][i_w1]
    #                   for i_f1 in range(NB_POINTS))) for i_w1 in range(NB_POINTS)]
    # pdf_f1_c1 = [sum((image_w1_f1_1[i_f1][i_w1]
    #                   for i_w1 in range(NB_POINTS))) for i_f1 in range(NB_POINTS)]
    # Construct PDF for sum white and flicker:
    s_min = w_1s[0] - d_w1 / 2 + f_1s[0] - d_f1 / 2
    s_max = w_1s[-1] + d_w1 / 2 + f_1s[-1] + d_f1 / 2
    d_s1 = (s_max - s_min) / NB_POINTS
    s_1s = cast(List[float], np.linspace(s_min + d_s1 / 2, s_max - d_s1 / 2,
                                        NB_POINTS, endpoint=True))
    pdf_s1_c1: List[float] = [0] * NB_POINTS
    d_s = min(d_f1, d_w1)
    d_l = max(d_f1, d_w1)
    for p_row, f_1i in zip(image_w1_f1_1, f_1s):
        f_1i_0 = f_1i - d_f1 / 2
        f_1i_1 = f_1i + d_f1 / 2
        for p, w_1i in zip(p_row, w_1s):
            w_1i_0 = w_1i - d_w1 / 2
            w_1i_1 = w_1i + d_w1 / 2
            s_0 = w_1i_0 + f_1i_0
            s_3 = w_1i_1 + f_1i_1
            s_1 = min(w_1i_0 + f_1i_1, w_1i_1 + f_1i_0)
            s_2 = max(w_1i_0 + f_1i_1, w_1i_1 + f_1i_0)
            for i_s, s_1i in enumerate(s_1s):
                s_1i_0 = s_1i - d_s1 / 2
                s_1i_1 = s_1i + d_s1 / 2
                if s_1i_1 <= s_0:
                    continue
                if s_1i_0 >= s_3:
                    continue
                # Lower triangle
                f_0 = (min(max(s_1i_0, s_0), s_1) - s_0) / (s_1 - s_0)
                f_1 = (max(min(s_1i_1, s_1), s_0) - s_0) / (s_1 - s_0)
                pdf_s1_c1[i_s] += d_s**2 / 2 * (f_1**2 - f_0**2) * p / d_s1
                # Higher triangle
                f_0 = (min(max(s_1i_0, s_2), s_3) - s_2) / (s_3 - s_2)
                f_1 = (max(min(s_1i_1, s_3), s_2) - s_2) / (s_3 - s_2)
                pdf_s1_c1[i_s] += d_s**2 * (f_1 - f_0 - f_1**2 / 2 + f_0**2 / 2) * p / d_s1
                # Middle parallel
                f_0 = (min(max(s_1i_0, s_1), s_2) - s_1) / (s_2 - s_1)
                f_1 = (max(min(s_1i_1, s_2), s_1) - s_1) / (s_2 - s_1)
                pdf_s1_c1[i_s] += d_s * (d_l - d_s) * (f_1 - f_0) * p / d_s1

    if args.v:
        print('PDF sanity check:')
        print(sum(sum(sum(sum(pdf_array_bit_1)))) * d_w0 * d_w1 * d_f0 * d_f1) # type: ignore
        print(sum((sum((image_w1_f1_1[i_f1][i_w1] for i_w1 in range(NB_POINTS)))
                for i_f1 in range(NB_POINTS))) * d_f1 * d_w1)
        print(sum(pdf_s1_c1) * d_s1)

    NB_PHASE_SHIFT = 1000
    phase_shifts = cast(List[float], np.linspace(0, 2 * np.pi, NB_PHASE_SHIFT, endpoint=True))
    p_1_m = 0.0 # pylint: disable=invalid-name
    sh_m = 0.0 # pylint: disable=invalid-name
    for sh_i in phase_shifts:
        p_1_i = 0 # pylint: disable=invalid-name
        for s_1i, p in zip(s_1s, pdf_s1_c1):
            if int(np.floor((s_1i + nom_p_1 + sh_i) / np.pi)) % 2 == 1:
                p_1_i += p * d_s1
        if p_1_i > p_1_m:
            p_1_m = p_1_i
            sh_m = sh_i
    data_to_store: List[List[float]] = [
        [nom_p_1, d_s1, p_1_m, sh_m],
        s_1s,
        pdf_s1_c1
    ]
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
    nom_p_1, d_s1, p_1_m, sh_m = data_from_store[0]
    s_1s = data_from_store[1]
    pdf_s1_c1 = data_from_store[2]

if args.v:
    p_1 = 0 # pylint: disable=invalid-name
    for s_1i, p in zip(s_1s, pdf_s1_c1):
        if int(np.floor((s_1i + nom_p_1) / np.pi)) % 2 == 1:
            p_1 += p * d_s1
    print(f'Probability bit equals 1: {p_1*100:.3f} %.')
    print(f'Shannon entropy: {-p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1)} bit.')


if args.v:
    print(f'Worst probability bit equals 1: {p_1_m*100:.3f} %.')
    print(f'Worst Shannon entropy: '
        f'{-p_1_m * np.log2(p_1_m) - (1 - p_1_m) * np.log2(1 - p_1_m)} bit.')

xs = [s_1i + nom_p_1 for s_1i in s_1s]
fill_where = [int((xi + sh_m) / np.pi) % 2 == 1 for xi in xs]

# Shift graph a bit to the left to make the legend fit:
x_shift = np.pi
x_min = min(xs)
x_max = max(xs)
x_shifted: List[float] = []
y_shifted: List[float] = []
for xi, yi in zip(xs, pdf_s1_c1):
    if xi >= x_min + x_shift:
        x_shifted.append(xi)
        y_shifted.append(yi)
fill_where_shifted = [int((xi + sh_m) / np.pi) % 2 == 1 for xi in x_shifted]

graph_maker = g_m.GraphMaker('sample_two_worst.svg', folder_name='figures')
graph_maker.create_grid()
ax = graph_maker.create_ax(title='Oscillator phase PDF', # pylint: disable=invalid-name
                           x_label=r'Oscillator phase ($\varphi$)',
                           y_label='Probability density', x_unit='rad',
                           y_unit=r'rad\textsuperscript{-1}',
                           x_scale='pi',
                           show_legend=True)
graph_maker.plot(ax, x_shifted, y_shifted,
                 label=r"""$f_{\Phi(t_2) \mid B'_1}(\varphi \mid 1)$""")
graph_maker.fill_between_y(ax, x_shifted, y_shifted, where=fill_where_shifted,
                           label=r"""$B'_{worst, 2}(\delta^{m}_{\phi}) = 1 \mid B'_1 = 1$""")
graph_maker.write_svg()
