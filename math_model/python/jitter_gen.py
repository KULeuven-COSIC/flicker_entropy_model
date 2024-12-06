"""Simulate a jitter measurement with a single RO. Measure accumulated phase
for nb_exp times at the given acc_times time points."""

import sys
from os import getcwd
from os.path import join
from typing import List, Optional, cast, Tuple, Type
import matplotlib.pyplot as plt
from scipy import optimize # type: ignore
import numpy as np
sys.path.append(getcwd())
from math_model.python import noises # pylint: disable=wrong-import-position
from math_model.python import data_reader as d_r # pylint: disable=wrong-import-position
from lib import time_logger as t_l # pylint: disable=wrong-import-position

class JitterGen:
    """Jitter measurement simulation class."""

    default_db_folder = join('math_model', 'simulation_data', 'jitter_gen')

    def __init__(self, gen_data: 'noises.GenerateData', acc_times: List[float],
                 db_folder: Optional[str]=None, verbose: bool=True):
        self._gen_data = gen_data
        self._acc_times = acc_times
        if db_folder is None:
            self._db_folder = JitterGen.default_db_folder
        else:
            self._db_folder = db_folder
        self._jit_man = JitterManager(self._acc_times, self._gen_data,
                                      self._db_folder, verbose)

    @property
    def jit_man(self) -> 'JitterManager':
        """This jittergen's jitter manager."""
        return self._jit_man

    def generate(self, nb_exp: int) -> None:
        """Generate nb_exp phase values at this jittergen's accumulation
        times. Store the resulting statistics in the db_folder."""
        nb_acc_times: int = len(self._acc_times)
        logger = t_l.TimeLogger(nb_exp)
        logger.start()
        for _ in range(nb_exp):
            phases: List[float] = [0] * nb_acc_times
            for ti_index, ti in enumerate(self._acc_times):
                phase_gen = noises.PhaseGenInst.generate_0(self._gen_data)
                phase_gen.add_time_points([ti]) # type: ignore
                phases[ti_index] = phase_gen.phase[1] # type: ignore
            self._jit_man.add_data(phases)
            logger.iterate()
        logger.clear()

class JitterManager:
    """A class containing data managing functionality for jitter measurements."""

    def __init__(self, acc_times: List[float], gen_data: 'noises.GenerateData',
                 db_folder: Optional[str]=None, verbose: bool=True):
        self._acc_times = acc_times
        self._gen_data = gen_data
        self._nom_phases = [2 * np.pi * self._gen_data.fn * ti for ti in self._acc_times]
        self._data_man = d_r.DataManager(self.file_name, db_folder)
        acc_time_str: List[str] = [f'{a:9.3e}' for a in self._acc_times]
        self._data_man.init_file(['Accumulation times [s]'] + acc_time_str,
                                 verbose=verbose)

    @classmethod
    def generate(cls: Type['JitterManager'], db_folder: Optional[str]=None) \
        -> List['JitterManager']:
        """Generate a list of jitter managers, for each file in the db_folder."""
        if db_folder is None:
            db_folder = JitterGen.default_db_folder
        data_managers = d_r.DataManager.generate(db_folder)
        result: List['JitterManager'] = []
        for dm in data_managers:
            header = dm.get_header()
            assert header is not None
            acc_times = header[1:]
            _, _, _, fn, fl, fh, hs, ncs \
                = JitterManager.parse_file_name(dm.file_name)
            gen_data = noises.GenerateData(fn, noises.Noise.FreqBound(fl, fh),
                                           hs, ncs)
            result.append(JitterManager(acc_times, gen_data, db_folder, False))
        return result

    @property
    def file_name(self) -> str:
        """This jitter manager's file name."""
        noise_titles: str = ','.join([cast(str, nt.short_title)
                                      for nt in self._gen_data.noise_classes])
        return '_'.join((
            'jitter',
            f'ac-mi:{min(self._acc_times):9.3e}',
            f'ac-ma:{max(self._acc_times):9.3e}',
            f'ac-nb:{len(self._acc_times)}',
            f'fn:{self._gen_data.fn:9.3e}',
            f'fl:{self._gen_data.freq_bound.fl:9.3e}',
            f'fh:{self._gen_data.freq_bound.fh:9.3e}',
            'hs:' + ','.join([f'{h:9.3e}' for h in self._gen_data.hs]),
            f'nts:{noise_titles}'
        )) + '.csv'

    @staticmethod
    def parse_file_name(file_name: str) \
        -> Tuple[float, float, int, float, float, float, List[float], List[Type['noises.Noise']]]:
        """Parse the given file_name."""
        parts = file_name.split('_')
        ac_mi = float(parts[1].split(':')[-1])
        ac_ma = float(parts[2].split(':')[-1])
        ac_nb = int(parts[3].split(':')[-1])
        fn = float(parts[4].split(':')[-1])
        fl = float(parts[5].split(':')[-1])
        fh = float(parts[6].split(':')[-1])
        hs = [float(p) for p in parts[7].split(':')[-1].split(',')]
        nts = parts[8].split(':')[-1].split(',')
        nts_c: List[Type['noises.Noise']] = []
        for n in nts:
            if n == 'wfm':
                nts_c.append(noises.WhiteFMNoise)
            else:
                nts_c.append(noises.FlickerFMNoise)
        return (ac_mi, ac_ma, ac_nb, fn, fl, fh, hs, nts_c)

    def add_data(self, phases: List[float]) -> None:
        """Add the given list of phases to this manager's database file."""
        phases_str: List[str] = [f'{p - np:9.3e}' for p, np in zip(phases, self._nom_phases)]
        self._data_man.append_row(phases_str)

    def get_data(self) -> List[List[float]]:
        """Get the stored phases."""
        raw_data = cast(List[List[float]], self._data_man.get_data())
        for exp in raw_data:
            for i, (p, np_i) in enumerate(zip(exp, self._nom_phases)):
                exp[i] = p + np_i
        return raw_data

    def plot_data(self) -> None:
        """Plot the data stored in this manager's database file."""
        data = self.get_data()
        print(f'Found {len(data)} experiments.')
        nb_accs = len(self._acc_times)
        phase_vars: List[float] = [0] * nb_accs
        for phase_index in range(nb_accs):
            phase_vars[phase_index] = np.var([exp[phase_index] # type: ignore
                                              for exp in data])
        _, axs = plt.subplots(2,1) # type: ignore
        for index, t in enumerate(self._acc_times):
            quant = np.percentile([exp[index]-2*np.pi*self._gen_data.fn*t # type: ignore
                                 for exp in data], 90)
            axs[0].plot(t, quant, 'o', color='grey', alpha=0.25) # type: ignore
        axs[0].plot(self._acc_times, # type: ignore
                    [2*np.pi*self._gen_data.fn*t for t in self._acc_times],
                    color='red', label='nominal phase (2pif_nt)')
        axs[0].set_xscale('log', base=10) # type: ignore
        axs[0].set_yscale('log', base=10) # type: ignore
        axs[0].set_title('Generated excess phases 90% quantile') # type: ignore
        axs[0].set_xlabel('Accumulation time [s]') # type: ignore
        axs[0].set_ylabel('Accumulated excess phase [rad]') # type: ignore
        axs[0].legend() # type: ignore
        axs[1].plot(self._acc_times, phase_vars, 'o', color='grey', alpha=0.25) # type: ignore
        theo_vars = self.calc_theoretical_phase_vars()
        for theo_var, noise_class in zip(theo_vars, self._gen_data.noise_classes):
            axs[1].plot(self._acc_times, theo_var, alpha=1, # type: ignore
                        label=noise_class.short_title)
        noise_corners = self.calc_noise_corners(theo_vars)
        for corner in noise_corners:
            print(f'{corner[0]}: {corner[1]:9.3e} s, {corner[2]:9.3e} rad^2')
            axs[1].plot(corner[1], corner[2], 'o', label=corner[0]) # type: ignore
        axs[1].set_xscale('log', base=10) # type: ignore
        axs[1].set_yscale('log', base=10) # type: ignore
        axs[1].set_title('Accumulated phase variance') # type: ignore
        axs[1].set_xlabel('Accumulation time [s]') # type: ignore
        axs[1].set_ylabel('Phase variance [rad^2]') # type: ignore
        axs[1].legend() # type: ignore
        plt.show() # type: ignore

    _sample_colors: List[str] = ['lightblue', 'bisque', 'lightgreen', 'lightpink', 'lightcoral']
    _theo_colors: List[str] = ['deepskyblue', 'orange', 'lawngreen', 'hotpink', 'coral']
    _corner_markers: List[str] = ['s', 'D', '8', 'P']

    @staticmethod
    def plot_all(db_folder: Optional[str]=None) -> None:
        """Plot all data available in the db_folder."""
        jit_mans = JitterManager.generate(db_folder)
        _, axs = plt.subplots(2,1) # type: ignore
        for jit_man, s_color, t_color in zip(jit_mans, JitterManager._sample_colors,
                                             JitterManager._theo_colors):
            data = jit_man.get_data()
            hf = jit_man._gen_data.hs[jit_man._gen_data.noise_classes.index(noises.FlickerFMNoise)] # pylint: disable=protected-access
            hw = jit_man._gen_data.hs[jit_man._gen_data.noise_classes.index(noises.WhiteFMNoise)] # pylint: disable=protected-access
            print(f'fn: {jit_man._gen_data.fn}, hf: {hf}, hw: {hw}, {len(data)} experiments.') # pylint: disable=protected-access
            nb_accs = len(jit_man._acc_times) # pylint: disable=protected-access
            phase_vars: List[float] = [0] * nb_accs
            for phase_index in range(nb_accs):
                phase_vars[phase_index] = np.var([exp[phase_index] # type: ignore
                                                  for exp in data])
            axs[1].plot(jit_man._acc_times, phase_vars, marker='o', color=s_color, # type: ignore # pylint: disable=protected-access
                        alpha=0.5,
                        label=f'fn: {jit_man._gen_data.fn:9.3e}, hf: {hf:9.3e}, hw: {hw:9.3e}') # pylint: disable=protected-access
            theo_vars = jit_man.calc_theoretical_phase_vars()
            for theo_var, noise_class in zip(theo_vars, jit_man._gen_data.noise_classes): # pylint: disable=protected-access
                line_style = 'solid' if noise_class.short_title == 'wfm' else 'dashed'
                label_h = f'hw: {hw:9.3e}' if noise_class.short_title == 'wfm' else f'hf: {hf:9.3e}'
                axs[1].plot(jit_man._acc_times, theo_var, alpha=1, # type: ignore # pylint: disable=protected-access
                            label=f'{noise_class.short_title}, {label_h}',
                            color=t_color, linestyle=line_style)
        for jit_man, t_color in zip(jit_mans, JitterManager._theo_colors):
            theo_vars = jit_man.calc_theoretical_phase_vars()
            noise_corners = jit_man.calc_noise_corners(theo_vars)
            for corner, c_marker in zip(noise_corners, JitterManager._corner_markers):
                print(f'{corner[0]}: {corner[1]:9.3e} s, {corner[2]:9.3e} rad^2')
                axs[1].plot(corner[1], corner[2], marker=c_marker, # type: ignore
                            label=f'{corner[0]} {corner[1]:9.3e} s', color=t_color)
        axs[1].set_xscale('log', basex=10) # type: ignore
        axs[1].set_yscale('log', basey=10) # type: ignore
        axs[1].set_title('Accumulated phase variance') # type: ignore
        axs[1].set_xlabel('Accumulation time [s]') # type: ignore
        axs[1].set_ylabel('Phase variance [rad^2]') # type: ignore
        axs[1].legend() # type: ignore
        plt.show()

    def calc_theoretical_phase_vars(self) -> List[List[float]]:
        """Calculate the theoretical phase variance for this manager's acc time
        points. A list of phase varainces for each noise seperately is generated."""
        result: List[List[float]] = []
        for noise_class, h in zip(self._gen_data.noise_classes, self._gen_data.hs):
            noise = noise_class.generate(self._gen_data.fn, h, self._gen_data.freq_bound)
            vars_: List[float] = [0.0] * len(self._acc_times)
            for index, ti in enumerate(self._acc_times):
                vars_[index] = noise.auto_cor(ti, ti)
            result.append(vars_)
        return result

    def calc_noise_corners(self, phase_vars: List[List[float]]) -> List[Tuple[str, float, float]]:
        """Calculate the theoretical noise corners for all noise type pairs
        involved."""
        result: List[Tuple[str, float, float]] = []
        for index0, noise_class0 in enumerate(self._gen_data.noise_classes):
            for index1, noise_class1 in enumerate(self._gen_data.noise_classes[index0+1:]):
                index1 += (index0 + 1)
                corner_name = cast(str, ' '.join([noise_class0.short_title, # type: ignore
                                                  noise_class1.short_title,
                                                  'corner']))
                vars0 = phase_vars[index0]
                vars1 = phase_vars[index1]
                var_diff = [v0-v1 for v0, v1 in zip(vars0, vars1)]
                crossings: List[int]
                crossings = cast(List[int], np.where(np.diff(np.sign(var_diff)))[0]) # type: ignore
                if len(crossings) > 0:
                    cr = crossings[0]
                    d0 = var_diff[cr]
                    d1 = var_diff[cr+1]
                    t0 = self._acc_times[cr]
                    t1 = self._acc_times[cr+1]
                    frac = cast(float, np.abs(d0)/np.abs(d1-d0)) # type: ignore
                    corner_time = frac * (t1 - t0) + t0
                    corner_var = frac * (vars0[cr+1] - vars0[cr]) + vars0[cr]
                    result.append((corner_name, corner_time, corner_var))
        return result

    def _calc_theo_fw_corner(self) -> float:
        """Calculate the theoretical flicker - white noise corner."""
        if noises.FlickerFMNoise not in self._gen_data.noise_classes:
            print('Error, no flicker noise present.')
            return -1
        hf = self._gen_data.hs[self._gen_data.noise_classes.index(noises.FlickerFMNoise)]
        if noises.WhiteFMNoise not in self._gen_data.noise_classes:
            print('Error, no white noise present.')
            return -1
        hw = self._gen_data.hs[self._gen_data.noise_classes.index(noises.WhiteFMNoise)]
        def f(t: float) -> float:
            return hw - hf * t * (3 - 2 * np.euler_gamma \
                                  - 2 * np.log(2 * np.pi * self._gen_data.freq_bound.fl * t))
        return optimize.root(f, 40e-9).x[0]
