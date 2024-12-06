"""Phase distribution estimator using a Monte Carlo method."""

import sys
from os import getcwd
from typing import List, cast, Tuple
import matplotlib.pyplot as plt # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore
import numpy as np
sys.path.append(getcwd())
from math_model.python import noises # pylint: disable=wrong-import-position
from math_model.python import data_reader as d_r # pylint: disable=wrong-import-position
from lib import time_logger as t_l # pylint: disable=wrong-import-position

class MCEstimator:
    """A class calculating the multivariate distribution for the flicker and
    white noise phases, using a Monte Carlo method. The TRNG is not reset in
    between bit generation."""

    def __init__(self, generate_data: 'noises.GenerateData',
                 nb_bits: int, sample_period: float):
        self._gen_data = generate_data
        self._nb_bits = nb_bits
        self._sample_period = sample_period
        h_w, h_f = 0.0, 0.0
        for h, noise_class in zip(self._gen_data.hs, self._gen_data.noise_classes):
            if noise_class.short_title == noises.WhiteFMNoise.short_title: # pylint: disable=comparison-with-callable
                h_w = h
            if noise_class.short_title == noises.FlickerFMNoise.short_title: # pylint: disable=comparison-with-callable
                h_f = h
        self._wn = noises.WhiteFMNoise(self._gen_data.fn, h_w, self._gen_data.freq_bound)
        self._fn = noises.FlickerFMNoise(self._gen_data.fn, h_f, self._gen_data.freq_bound)
        self._ts = [i * self._sample_period for i in range(self._nb_bits + 1)]
        self._nom_phases = [2 * np.pi * self._gen_data.fn * ti for ti in self._ts]
        self._s_w = self._wn.kernel(self._ts) # type: ignore
        self._s_f = self._fn.kernel(self._ts) # type: ignore
        self._m_w = np.zeros((self._nb_bits + 1, 1))
        self._m_f = np.zeros((self._nb_bits + 1, 1))

    def _get_samples(self, nb_samples: int) -> Tuple[List[List[float]], List[List[float]]]:
        samples_w = cast(List[List[float]], np.random.multivariate_normal(self._m_w.flat,
                                                                          self._s_w,
                                                                          nb_samples))
        samples_w = cast(List[List[float]],
                         [list(samples.flat) for samples in samples_w]) # type: ignore
        samples_f = cast(List[List[float]], np.random.multivariate_normal(self._m_f.flat,
                                                                          self._s_f,
                                                                          nb_samples))
        samples_f = cast(List[List[float]],
                         [list(samples.flat) for samples in samples_f]) # type: ignore
        return (samples_w, samples_f)

    def _est_mc_prob(self, bits: List[int], nb_mc_samples: int) \
        -> Tuple[float, List[List[float]], List[List[float]], List[bool]]:
        if len(bits) < self._nb_bits:
            bits += [-1] * (self._nb_bits - len(bits))
        samples_w, samples_f = self._get_samples(nb_mc_samples)
        tot_phases = [[n + w + f for n, w, f in zip(sample_w, sample_f, self._nom_phases)]
                      for sample_w, sample_f in zip(samples_w, samples_f)]
        gen_bits = [[int(p / np.pi) % 2 for p in phases][1:] for phases in tot_phases]
        correct: List[bool] = [True] * nb_mc_samples
        for index, gen_bit in enumerate(gen_bits):
            all_correct = True
            for bit_g, bit_c in zip(gen_bit, bits):
                if bit_c >= 0:
                    if bit_c != bit_g:
                        all_correct = False
                        break
            correct[index] = all_correct
        return (sum(correct) / nb_mc_samples,
                samples_w, samples_f,
                correct)

    def _est_mc_entr(self, samples_w: List[List[float]], samples_f: List[List[float]],
                     correct: List[bool]) -> List[float]:
        result: List[float] = [0] * (self._nb_bits + 1)
        for sample_w, sample_f, cor in zip(samples_w, samples_f, correct):
            if cor:
                for index, (w, f, n) in enumerate(zip(sample_w, sample_f, self._nom_phases)):
                    phase = n + w + f
                    bit = int(phase / np.pi) % 2
                    result[index] += bit
        nb_correct = sum(correct)
        result = [r / nb_correct for r in result]
        return [cast(float, -np.log2(max(r, 1-r))) for r in result] # type: ignore

    def est_mc_dist_evolve(self, nb_mc_samples: int,
                            max_depth_single: int, max_depth_comb: int) \
        -> Tuple[List[float], List[float],
                 List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
        """Get the final bit phase samples and value of the given history bits, both for
        a single history bit known, as a combination of history bits known."""
        indices_single = [[i] for i in range(-2, -2 - max_depth_single, -1)]
        indices_comb = [list(range(-2, -2 - i, -1)) for i in range(1, max_depth_comb + 1)]
        final_w, final_f, hist, _ = self._est_mc_dist_evolve_ind(nb_mc_samples,
                                                              indices_single + indices_comb)
        return (final_w, final_f, hist[:max_depth_single], hist[max_depth_single:],
                indices_single, indices_comb)

    def _est_mc_dist_evolve_ind(self, nb_mc_samples: int, indices: List[List[int]]) \
        -> Tuple[List[float], List[float], List[List[int]], List[List[float]]]:
        samples_w, samples_f = self._get_samples(nb_mc_samples)
        bits = [[int((w + f + n) / np.pi) % 2
                 for w, f, n in zip(sample_w, sample_f, self._nom_phases)]
                 for sample_w, sample_f in zip(samples_w, samples_f)]
        result: List[List[int]] = []
        for index_group in indices:
            result_group: List[int] = [0] * nb_mc_samples
            for sample_index, sample_bits in enumerate(bits):
                sel_bits = (sample_bits[index] for index in index_group)
                value = sum((b * 2**i for i, b in enumerate(sel_bits)))
                result_group[sample_index] = value
            result.append(result_group)
        final_w = [sample_w[-1] for sample_w in samples_w]
        final_f = [sample_f[-1] for sample_f in samples_f]
        tot_phase = [[f + w for f, w in zip(sample_w, sample_f)]
                     for sample_w, sample_f in zip(samples_w, samples_f)]
        return (final_w, final_f, result, tot_phase)

    def plot_mc_dist_evolve(self, nb_mc_samples: int,
                            indices_single: List[int],
                            # indices_comb: List[List[int]],
                            nb_bins: int=50) -> None:
        """Calculate bit history and select from history, plot evolving final
        bit distribution."""
        final_w, final_f, single_hist, tot_phase \
            = self._est_mc_dist_evolve_ind(nb_mc_samples, [[i] for i in indices_single])
        fig = plt.figure(layout="constrained") # type: ignore
        nb_hor_figs = int(np.ceil(len(indices_single) / 2)) * 2
        gs = GridSpec(3, nb_hor_figs, figure=fig)
        # Plot single history:
        ax_single_0 = fig.add_subplot(gs[0,:int(nb_hor_figs / 2)]) # type: ignore
        ax_single_1 = fig.add_subplot(gs[0,int(nb_hor_figs / 2):]) # type: ignore
        last_hist_t, last_hist_y, _ \
            = MCEstimator._calc_hist([w + f for w, f in zip(final_w, final_f)],
                                                  nb_bins)
        min_t_0, max_t_0 = min(last_hist_t), max(last_hist_t)
        min_t_1, max_t_1 = min_t_0, max_t_0
        max_y_0, max_y_1 = max(last_hist_y), max(last_hist_y)
        prob_1 = sum((int((w + f + self._nom_phases[-1]) / np.pi) % 2
                      for w, f in zip(final_w, final_f))) / nb_mc_samples
        entr = -np.log2(max(prob_1, 1 - prob_1)) # type: ignore
        ax_single_0.plot(last_hist_t, last_hist_y, label=f'no hist, H={entr:4.2f}') # type: ignore
        ax_single_1.plot(last_hist_t, last_hist_y, label=f'no hist, H={entr:4.2f}') # type: ignore
        for index, hist_bit in enumerate(indices_single):
            sel_0 = [w + f for w, f, h in zip(final_w, final_f, single_hist[index]) if h == 0]
            sel_1 = [w + f for w, f, h in zip(final_w, final_f, single_hist[index]) if h == 1]
            ts_0, ys_0, _ = MCEstimator._calc_hist(sel_0, nb_bins)
            ts_1, ys_1, _ = MCEstimator._calc_hist(sel_1, nb_bins)
            if sel_0:
                min_temp_0, max_temp_0 = min(ts_0), max(ts_0)
                min_t_0, max_t_0 = min(min_t_0, min_temp_0), max(max_t_0, max_temp_0)
                max_temp_0 = max(ys_0)
                max_y_0 = max(max_y_0, max_temp_0)
                prob_1_0 = sum((int((p + self._nom_phases[-1]) / np.pi) % 2
                                for p in sel_0)) / len(sel_0)
            else:
                prob_1_0 = 1
            if sel_1:
                min_temp_1, max_temp_1 = min(ts_1), max(ts_1)
                min_t_1, max_t_1 = min(min_t_1, min_temp_1), max(max_t_1, max_temp_1)
                max_temp_1 = max(ys_1)
                max_y_1 = max(max_y_1, max_temp_1)
                prob_1_1 = sum((int((p + self._nom_phases[-1]) / np.pi) % 2
                                for p in sel_1)) / len(sel_1)
            else:
                prob_1_1 = 1
            entr_0 = -np.log2(max(prob_1_0, 1 - prob_1_0)) # type: ignore
            entr_1 = -np.log2(max(prob_1_1, 1 - prob_1_1)) # type: ignore
            ax_single_0.plot(ts_0, ys_0, label=f'hist {hist_bit+1}, H={entr_0:4.2f}') # type: ignore
            ax_single_1.plot(ts_1, ys_1, label=f'hist {hist_bit+1}, H={entr_1:4.2f}') # type: ignore
        nom_p = self._nom_phases[-1]
        pi_mult_0 = [n * np.pi - nom_p
                     for n in range(int(np.ceil((min_t_0 + nom_p) / np.pi)),
                                    int(np.floor((max_t_0 + nom_p) / np.pi)) + 1)] # type: ignore
        pi_mult_1 = [n * np.pi - nom_p
                     for n in range(int(np.ceil((min_t_1 + nom_p) / np.pi)),
                                    int(np.floor((max_t_1 + nom_p) / np.pi)) + 1)] # type: ignore
        ax_single_0.vlines(pi_mult_0, 0, max_y_0, colors='grey', # type: ignore
                           linestyles='dotted', label='pi mult')
        ax_single_1.vlines(pi_mult_1, 0, max_y_1, colors='grey', # type: ignore
                           linestyles='dotted', label='pi mult')
        ax_single_0.set_title(f'History 0 bit {self._nb_bits} phase histogram') # type: ignore
        ax_single_0.set_xlabel('Excess phase [rad]') # type: ignore
        ax_single_0.set_ylabel('Relative occurence [-]') # type: ignore
        ax_single_0.legend(loc='upper right') # type: ignore
        ax_single_1.set_title(f'History 1 bit {self._nb_bits} phase histogram') # type: ignore
        ax_single_1.set_xlabel('Excess phase [rad]') # type: ignore
        ax_single_1.set_ylabel('Relative occurence [-]') # type: ignore
        ax_single_1.legend(loc='upper right') # type: ignore
        # Plot each single history bit distribution sperately:
        axs_single = []
        for index, hist_bit in enumerate(indices_single):
            axs_single.append(fig.add_subplot(gs[1,index])) # type: ignore
            sel_0 = [w + f for w, f, h in zip(final_w, final_f, single_hist[index]) if h == 0]
            sel_1 = [w + f for w, f, h in zip(final_w, final_f, single_hist[index]) if h == 1]
            ts_0, ys_0, _ = MCEstimator._calc_hist(sel_0, nb_bins)
            ts_1, ys_1, _ = MCEstimator._calc_hist(sel_1, nb_bins)
            min_t = min(*ts_0, *ts_1)
            max_t = max(*ts_0, *ts_1)
            max_y = max(*ys_0, *ys_1)
            axs_single[-1].plot(ts_0, ys_0, label='hist 0') # type: ignore
            axs_single[-1].plot(ts_1, ys_1, label='hist 1') # type: ignore
            pi_mult = [n * np.pi - nom_p
                       for n in range(int(np.ceil((min_t + nom_p) / np.pi)),
                                      int(np.floor((max_t + nom_p) / np.pi)) + 1)] # type: ignore
            axs_single[-1].vlines(pi_mult, 0, max_y, colors='grey', # type: ignore
                                  linestyles='dotted', label='pi mult')
            axs_single[-1].set_title(('History bit ' # type: ignore
                                      f'{hist_bit + 1} phase histogram'))
            axs_single[-1].set_xlabel('Excess phase [rad]') # type: ignore
            axs_single[-1].set_ylabel('Relative occurence [-]') # type: ignore
            axs_single[-1].legend(loc='upper right') # type: ignore
        # Plot all phases for the first single index group:
        ax_phase = fig.add_subplot(gs[2,:]) # type: ignore
        first_0, first_1 = True, True
        step = int(np.ceil(nb_mc_samples / 100)) # In case of too many plots
        plot_ran = range(0, nb_mc_samples, step)
        plot_data = ((single_hist[0][i], tot_phase[i]) for i in plot_ran)
        for value, phases in plot_data:
            kwargs = {
                'marker': 'o',
                'alpha': 0.1
            }
            kwargs['color'] = 'red' if value else 'grey'
            kwargs['alpha'] = 0.5 if value else 0.1
            if value:
                if first_1:
                    first_1 = False
                    kwargs['label'] = f'Excess phase, bit {indices_single[0] + 1} was 1'
            else:
                if first_0:
                    first_0 = False
                    kwargs['label'] = f'Excess phase, bit {indices_single[0] + 1} was 0'
            ax_phase.plot(self._ts, phases, **kwargs) # type: ignore
        ax_phase.set_title(('Generated phase for history ' # type: ignore
                            f'bit {indices_single[0] + 1}'))
        ax_phase.set_xlabel('Sampling time [s]') # type: ignore
        ax_phase.set_ylabel('Excess phase [rad]') # type: ignore
        ax_phase.legend(loc='upper right') # type: ignore
        plt.show() # type: ignore

    @staticmethod
    def _calc_hist(data: List[float], nb_bins: int=50) \
        -> Tuple[List[float], List[float], float]:
        nb_data = len(data)
        if nb_data == 0:
            return ([], [], 0)
        d_min = min(data)
        d_max = max(data)
        ts = cast(List[float], list(np.linspace(d_min, d_max, nb_bins).flat))
        bin_width = (d_max - d_min) / (nb_bins - 1)
        ys: List[float] = [0] * nb_bins
        for bin_ in range(nb_bins):
            bin_start = ts[bin_] - bin_width / 2
            bin_end = ts[bin_] + bin_width / 2
            for d in data:
                if (d >= bin_start) & (d < bin_end):
                    ys[bin_] += 1
            ys[bin_] /= nb_data
        return (ts, ys, bin_width)

    def plot_mc_prob(self, bits: List[int], nb_mc_samples: int) -> None:
        """Calculate the probability of obtaining the given bits, plot the
        result."""
        if len(bits) < self._nb_bits:
            bits += [-1] * (self._nb_bits - len(bits))
        prob, samples_w, samples_f, correct = self._est_mc_prob(bits, nb_mc_samples)
        print(f'Probability = {prob*100:9.3f} %')
        final_w_c = [sample[-1] for cor, sample in zip(correct, samples_w) if cor]
        final_f_c = [sample[-1] for cor, sample in zip(correct, samples_f) if cor]
        fig = plt.figure(layout="constrained") # type: ignore
        if self._nb_bits < 15: # We can plot all sampled bits here
            gs = GridSpec(2, 2, figure=fig)
            ax_hist = fig.add_subplot(gs[0,0]) # type: ignore
            ax_ent = fig.add_subplot(gs[0,1]) # type: ignore
            ax_samples = fig.add_subplot(gs[1,:]) # type: ignore
            ax_hist.hist([w + f for w, f in zip(final_w_c, final_f_c)], # type: ignore
                         color='green', alpha=0.25, label='sum')
            ax_hist.hist(final_w_c, # type: ignore
                         color='blue', alpha=0.25, label='white')
            ax_hist.hist(final_f_c, # type: ignore
                         color='orange', alpha=0.25, label='flicker')
            ax_hist.set_title('Selected phase histogram final bit') # type: ignore
            ax_hist.set_xlabel('Phase [rad]') # type: ignore
            ax_hist.set_ylabel('Occurence [-]') # type: ignore
            ax_hist.legend(loc='upper right') # type: ignore
            mc_entropy = self._est_mc_entr(samples_w, samples_f, correct)
            ax_ent.plot(self._ts, mc_entropy, marker='o') # type: ignore
            ax_ent.set_title('Selected phase min-entropy') # type: ignore
            ax_ent.set_xlabel('Sampling time [s]') # type: ignore
            ax_ent.set_ylabel('Min-entropy [bit]') # type: ignore
            min_p = min([min([min(w, f, w + f) for w, f in zip(sam_w, sam_f)])
                         for sam_w, sam_f in zip(samples_w, samples_f)])
            max_p = max([max([max(w, f, w + f) for w, f in zip(sam_w, sam_f)])
                         for sam_w, sam_f in zip(samples_w, samples_f)])
            for bit_i in range(self._nb_bits):
                nom_p = self._nom_phases[bit_i + 1]
                pi_n_min = int(np.floor((nom_p + min_p) / np.pi)) # type: ignore
                pi_n_max = int(np.ceil((nom_p + max_p) / np.pi)) # type: ignore
                if pi_n_min % 2 == 1:
                    pi_n_min += 1
                pi_values = [pi_n * np.pi - nom_p for pi_n in range(pi_n_min, pi_n_max + 1)]
                for index, (pi_value0, pi_value1) in enumerate(zip(pi_values[::2],
                                                                   pi_values[1::2])):
                    if (index == 0) & (bit_i == 0):
                        kwargs = {'label': 'bit = 0'}
                    else:
                        kwargs = {}
                    ts = [self._ts[bit_i + 1] - self._sample_period / 2,
                          self._ts[bit_i + 1] + self._sample_period / 2]
                    ax_samples.fill_between(ts, pi_value0 * np.ones(2), # type: ignore
                                            pi_value1 * np.ones(2), # type: ignore
                                            color='pink', alpha=0.35, **kwargs) # type: ignore
            for index, (sam_w, sam_f, cor) in enumerate(zip(samples_w, samples_f, correct)):
                sam_s = [w + f for w, f in zip(sam_w, sam_f)]
                kwargs = {
                    'marker': 'o',
                    'alpha': 0.75 if cor else 0.1, # type: ignore
                    'color': 'blue' if cor else 'lightblue'
                }
                if not index:
                    kwargs['label'] = 'white'
                # ax_samples.plot(self._ts, sam_w, **kwargs) # type: ignore
                kwargs['color'] = 'orange' if cor else 'bisque' # light orange
                if not index:
                    kwargs['label'] = 'flicker'
                # ax_samples.plot(self._ts, sam_f, **kwargs) # type: ignore
                kwargs['color'] = 'green' if cor else 'lightgreen'
                if not index:
                    kwargs['label'] = 'sum'
                ax_samples.plot(self._ts, sam_s, **kwargs) # type: ignore
            ax_samples.set_title('Phase samples') # type: ignore
            ax_samples.set_xlabel('Sampling time [s]') # type: ignore
            ax_samples.set_ylabel('Phase [rad]') # type: ignore
            ax_samples.legend(loc='upper right') # type: ignore
        plt.show() # type: ignore

class WFSweep:
    """Class for performing a sweep over h_flicker, sample_period, hist_type
    and save data in files."""

    default_fn: float = 520e6

    def __init__(self, fns: List[float], h_flickers: List[List[float]],
                 sample_pers: List[List[List[float]]],
                 nb_mc_samples: int, bit_nb: int,
                 mc_window: int=100000, nb_bins: int=1000, depth_single: int=10, depth_comb: int=5,
                 fl: float=1e-3, fh: float=10e9, hw: float=1.89e-14):
        self._fns = fns
        self._h_flickers = h_flickers
        self._sample_pers = sample_pers
        self._nb_mc_samples = nb_mc_samples
        self._bit_nb = bit_nb
        self._depth_single = depth_single
        self._depth_comb = depth_comb
        self._mc_window = mc_window
        self._nb_bins = nb_bins
        self._fl = fl
        self._fh = fh
        self._hw = hw
        self._nb_mc_windows = int(np.ceil(self._nb_mc_samples / self._mc_window))
        self._dist_man = d_r.DistributionManager()

    def sweep(self) -> None:
        """Iterate over all variables, store the distribution each mc_window."""
        nb_iter = len(self._fns) * len(self._h_flickers) \
            * sum((len(pers) for pers in self._sample_pers)) \
            * self._nb_mc_windows * (self._depth_single + self._depth_comb)
        logger = t_l.TimeLogger(nb_iter)
        logger.start()
        for fn_index, fn in enumerate(self._fns):
            for h_flicker_index, h_flicker in enumerate(self._h_flickers[fn_index]):
                for sample_per in self._sample_pers[fn_index][h_flicker_index]:
                    gen_kwargs = {
                        'fn': fn,
                        'freq_bound': noises.Noise.FreqBound(self._fl, self._fh),
                        'hs': [h_flicker, self._hw],
                        'noise_classes': [noises.FlickerFMNoise, noises.WhiteFMNoise],
                        'verbose': False
                    }
                    gen_data = noises.GenerateData(**gen_kwargs) # type: ignore
                    estimator = MCEstimator(gen_data, self._bit_nb, sample_per)
                    for _ in range(self._nb_mc_windows):
                        final_w, final_f, hist_single, hist_comb, ind_single, ind_comb \
                            = estimator.est_mc_dist_evolve(self._mc_window, self._depth_single,
                                                            self._depth_comb)
                        dist_id_w = d_r.DistributionManager.get_dist_id(self._bit_nb,
                                                                                [], 0, True)
                        dist_id_f = d_r.DistributionManager.get_dist_id(self._bit_nb,
                                                                                [], 0, False)
                        dist_w_s = self._dist_man.get_dist(dist_id_w, self._nb_bins, fn, h_flicker,
                                                        sample_per, 'single')
                        dist_f_s = self._dist_man.get_dist(dist_id_f, self._nb_bins, fn, h_flicker,
                                                        sample_per, 'single')
                        dist_w_c = self._dist_man.get_dist(dist_id_w, self._nb_bins, fn, h_flicker,
                                                        sample_per, 'comb')
                        dist_f_c = self._dist_man.get_dist(dist_id_f, self._nb_bins, fn, h_flicker,
                                                        sample_per, 'comb')
                        dist_w_s.add_samples(final_w)
                        dist_f_s.add_samples(final_f)
                        dist_w_c.add_samples(final_w)
                        dist_f_c.add_samples(final_f)
                        if self._depth_single:
                            self._dist_man.store_dist(dist_w_s, fn, h_flicker, sample_per, 'single')
                            self._dist_man.store_dist(dist_f_s, fn, h_flicker, sample_per, 'single')
                        if self._depth_comb:
                            self._dist_man.store_dist(dist_w_c, fn, h_flicker, sample_per, 'comb')
                            self._dist_man.store_dist(dist_f_c, fn, h_flicker, sample_per, 'comb')
                        for depth_single in range(self._depth_single):
                            bit_index = ind_single[depth_single][0] + self._bit_nb + 1
                            dist_id_0_w = d_r.DistributionManager.get_dist_id(self._bit_nb,
                                                                                    [bit_index],
                                                                                    0, True)
                            dist_id_0_f = d_r.DistributionManager.get_dist_id(self._bit_nb,
                                                                                    [bit_index],
                                                                                    0, False)
                            dist_id_1_w = d_r.DistributionManager.get_dist_id(self._bit_nb,
                                                                                    [bit_index],
                                                                                    1, True)
                            dist_id_1_f = d_r.DistributionManager.get_dist_id(self._bit_nb,
                                                                                    [bit_index],
                                                                                    1, False)
                            dist_0_w = self._dist_man.get_dist(dist_id_0_w, self._nb_bins, fn,
                                                               h_flicker, sample_per, 'single')
                            dist_0_f = self._dist_man.get_dist(dist_id_0_f, self._nb_bins, fn,
                                                               h_flicker, sample_per, 'single')
                            dist_1_w = self._dist_man.get_dist(dist_id_1_w, self._nb_bins, fn,
                                                               h_flicker, sample_per, 'single')
                            dist_1_f = self._dist_man.get_dist(dist_id_1_f, self._nb_bins, fn,
                                                               h_flicker, sample_per, 'single')
                            samples_0_w = [w for w, v in zip(final_w, hist_single[depth_single])
                                        if v == 0]
                            samples_0_f = [f for f, v in zip(final_f, hist_single[depth_single])
                                        if v == 0]
                            samples_1_w = [w for w, v in zip(final_w, hist_single[depth_single])
                                        if v == 1]
                            samples_1_f = [f for f, v in zip(final_f, hist_single[depth_single])
                                        if v == 1]
                            dist_0_w.add_samples(samples_0_w)
                            dist_0_f.add_samples(samples_0_f)
                            dist_1_w.add_samples(samples_1_w)
                            dist_1_f.add_samples(samples_1_f)
                            self._dist_man.store_dist(dist_0_w, fn, h_flicker, sample_per, 'single')
                            self._dist_man.store_dist(dist_0_f, fn, h_flicker, sample_per, 'single')
                            self._dist_man.store_dist(dist_1_w, fn, h_flicker, sample_per, 'single')
                            self._dist_man.store_dist(dist_1_f, fn, h_flicker, sample_per, 'single')
                            logger.iterate()
                        for depth_comb in range(self._depth_comb):
                            nb_hist_bits = depth_comb + 1
                            nb_hist_values = 2**nb_hist_bits
                            bit_indexes = [ic + self._bit_nb + 1 for ic in ind_comb[depth_comb]]
                            for hist_value in range(nb_hist_values):
                                id_w \
                                    = d_r.DistributionManager.get_dist_id(self._bit_nb,
                                                                                  bit_indexes,
                                                                                  hist_value, True)
                                id_f \
                                    = d_r.DistributionManager.get_dist_id(self._bit_nb,
                                                                                  bit_indexes,
                                                                                  hist_value, False)
                                dist_w = self._dist_man.get_dist(id_w, self._nb_bins, fn, h_flicker,
                                                                 sample_per, 'comb')
                                dist_f = self._dist_man.get_dist(id_f, self._nb_bins, fn, h_flicker,
                                                                 sample_per, 'comb')
                                sws = [w for w, v in zip(final_w, hist_comb[depth_comb])
                                    if v == hist_value]
                                sfs = [f for f, v in zip(final_f, hist_comb[depth_comb])
                                    if v == hist_value]
                                dist_w.add_samples(sws)
                                dist_f.add_samples(sfs)
                                self._dist_man.store_dist(dist_w, fn, h_flicker, sample_per, 'comb')
                                self._dist_man.store_dist(dist_f, fn, h_flicker, sample_per, 'comb')
                            logger.iterate()
        logger.clear()
