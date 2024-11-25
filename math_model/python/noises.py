"""This module contains the noise phase generation code."""

from typing import Optional, List, Type, TypeVar, Tuple, cast
from abc import ABC, abstractmethod
import warnings
import matplotlib.pyplot as plt
import scipy.special # type: ignore
import numpy as np
import numpy.random as npr

NT = TypeVar('NT', bound='Noise')
PT = TypeVar('PT', bound='PhaseGenInst')

class ClassProperty(property):
    """Custom class property."""

    def __get__(self, owner_self: object, owner_cls: Optional[type]=None):
        return self.fget(owner_cls) # type: ignore

class Noise(ABC):
    """
    Single noise source instance.
    """

    _title: str
    _short_title: str

    class FreqBound:
        """
        Noise frequency bounds.
        """

        def __init__(self, fl: float = 0, fh: float = 0) -> None:
            self._fl: float = fl
            self._fh: float = fh

        @property
        def fl(self) -> float:
            """Lower frequency bound."""
            return self._fl

        @property
        def fh(self) -> float:
            """Higher frequency bound."""
            return self._fh

    def __init__(self, type_: int, fn: float, h: float,
                 freq_bound: FreqBound) -> None:
        self.type: int = type_
        self.fn: float = fn
        self.h: float  = h
        self._freq_bound: Noise.FreqBound = freq_bound

    @classmethod
    @abstractmethod
    def generate(cls: Type[NT], fn: float, h: float, freq_bound: FreqBound) -> NT:
        """Generate a new noise source."""

    @ClassProperty
    def title(cls) -> str: # pylint: disable=no-self-argument
        """Noise source title."""
        return cls._title

    @ClassProperty
    def short_title(cls) -> str: # pylint: disable=no-self-argument
        """Noise source short title."""
        return cls._short_title

    @abstractmethod
    def auto_cor(self, t0: float, t1: float) -> float:
        """Get the auto-correlation value between times t0 and t1."""

    def kernel(self, t0s: List[float],
               t1s: Optional[List[float]]=None) \
        -> List[List[float]]:
        """Generate a kernel array for the given time instances."""
        nb_points_0: int = len(t0s)
        nb_points_1: int = 0
        result: List[List[float]]
        if t1s is None:
            result = np.zeros((nb_points_0, nb_points_0)) # type: ignore
        else:
            nb_points_1 = len(t1s)
            result = np.zeros((nb_points_0, nb_points_1)) # type: ignore
        if t1s is None:
            for t0i in range(nb_points_0):
                t0 = t0s[t0i] # type: ignore
                for t1i in range(t0i+1):
                    t1 = t0s[t1i] # type: ignore
                    result[t0i][t1i] = self.auto_cor(t0, t1) # type: ignore
                    result[t1i][t0i] = result[t0i][t1i] # type: ignore
        else:
            for t0i in range(nb_points_0):
                t0 = t0s[t0i] # type: ignore
                for t1i in range(nb_points_1):
                    t1 = t1s[t1i] # type: ignore
                    result[t0i][t1i] = self.auto_cor(t0, t1) # type: ignore
        return result

class WhiteFMNoise(Noise):
    """Single white FM noise source."""

    _title: str = 'White FM Noise'
    _short_title: str = 'wfm'

    def __init__(self, fn: float, h: float,
                 freq_bound: Noise.FreqBound=Noise.FreqBound(1e-3, 10e9)) -> None:
        super().__init__(0, fn, h, freq_bound)

    @classmethod
    def generate(cls, fn: float, h: float, freq_bound: Noise.FreqBound) \
        -> 'WhiteFMNoise':
        return cls(fn, h, freq_bound)

    def auto_cor(self, t0: float, t1: float) -> float:
        return 4 * np.pi**2 * self.fn**2 * self.h * min(t0, t1)

class FlickerFMNoise(Noise):
    """Single flicker FM noise source."""

    _title: str = 'Flicker FM Noise'
    _short_title: str = 'ffm'

    def __init__(self, fn: float, h: float,
                 freq_bound: Noise.FreqBound=Noise.FreqBound(1e-3, 10e9)) \
                    -> None:
        super().__init__(-1, fn, h, freq_bound)

    @classmethod
    def generate(cls, fn: float, h: float, freq_bound: Noise.FreqBound) \
        -> 'FlickerFMNoise':
        return cls(fn, h, freq_bound)

    def auto_cor(self, t0: float, t1: float) -> float:
        return self._auto_cor_simpl(t0, t1)

    def _auto_cor_exact(self, t0: float, t1: float) -> float:
        result: float = 0
        if (t0 != 0) | (t1 != 0):
            result += 2*self.fn**2*self.h*(
                - np.cos(2 * np.pi * self._freq_bound.fh * (t1 - t0))
                    / (2 * self._freq_bound.fh**2)
                + np.pi * (t1 - t0)
                    * np.sin(2 * np.pi * self._freq_bound.fh * (t1 - t0))
                    / self._freq_bound.fh
                + np.cos(2 * np.pi * self._freq_bound.fl * (t1 - t0))
                    / (2 * self._freq_bound.fl**2)
                - np.pi * (t1 - t0)
                    * np.sin(2 * np.pi * self._freq_bound.fl * (t1 - t0))
                    / self._freq_bound.fl
                + np.cos(2 * np.pi * self._freq_bound.fh * t1)
                    / (2 * self._freq_bound.fh**2)
                - np.pi * t1 * np.sin(2 * np.pi * self._freq_bound.fh * t1)
                    / self._freq_bound.fh
                - np.cos(2 * np.pi * self._freq_bound.fl * t1)
                    / (2 * self._freq_bound.fl**2)
                + np.pi * t1 * np.sin(2 * np.pi * self._freq_bound.fl * t1)
                    / self._freq_bound.fl
                + np.cos(2 * np.pi * self._freq_bound.fh * t0)
                    / (2 * self._freq_bound.fh**2)
                - np.pi * t0 * np.sin(2 * np.pi * self._freq_bound.fh * t0)
                    / self._freq_bound.fh
                - np.cos(2 * np.pi * self._freq_bound.fl * t0)
                    / (2 * self._freq_bound.fl**2)
                + np.pi * t0 * np.sin(2 * np.pi * self._freq_bound.fl * t0)
                    / self._freq_bound.fl
                + 1 / (2 * self._freq_bound.fl**2)
                - 1 / (2 * self._freq_bound.fh**2)
            )
        if t0 != 0:
            result += 2 * self.fn**2 * self.h * (
                # pylint: disable=no-member
                + 2 * np.pi**2 * t0**2
                    * scipy.special.sici(2 * np.pi # type: ignore
                                         * self._freq_bound.fh * t0)[1]
                # pylint: disable=no-member
                - 2 * np.pi**2 * t0**2
                    * scipy.special.sici(2 * np.pi # type: ignore
                                         * self._freq_bound.fl * t0)[1]
            )
        if t1 != 0:
            result += 2 * self.fn**2 * self.h * (
                # pylint: disable=no-member
                + 2 * np.pi**2 * t1**2
                    * scipy.special.sici(2 * np.pi # type: ignore
                                         * self._freq_bound.fh * t1)[1]
                # pylint: disable=no-member
                - 2 * np.pi**2 * t1**2
                    * scipy.special.sici(2 * np.pi # type: ignore
                                         * self._freq_bound.fl * t1)[1]
            )
        if t0 != t1:
            result += 2 * self.fn**2 * self.h * (
                # pylint: disable=no-member
                - 2 * np.pi**2 * (t1 - t0)**2
                    * scipy.special.sici(2 * np.pi # type: ignore
                                         * self._freq_bound.fh
                                         * abs(t1 - t0))[1]
                # pylint: disable=no-member
                + 2 * np.pi**2 * (t1 - t0)**2
                    * scipy.special.sici(2 * np.pi # type: ignore
                                         * self._freq_bound.fl
                                         * abs(t1 - t0))[1]
            )
        return result

    def _auto_cor_simpl(self, t0: float, t1: float) -> float:
        result: float = 0
        if t0 != t1:
            if (t0 != 0) & (t1 != 0):
                result = 4 * np.pi**2 * self.fn**2 * self.h * t0 * t1 * (
                    3 - 2 * np.euler_gamma
                    - 2 * np.log(2 * np.pi * self._freq_bound.fl * abs(t1 - t0))
                    + t0 / t1 * np.log(abs(t1 - t0) / t0)
                    + t1 / t0 * np.log(abs(t1 - t0) / t1)
                )
        else:
            if t0 != 0:
                result = 4 * np.pi**2 * self.fn**2 * self.h * t0**2 * (
                    3 - 2 * np.euler_gamma
                    - 2 * np.log(2 * np.pi * self._freq_bound.fl * t0)
                )
        return result

class GenerateData:
    """Generate data container."""
    def __init__(self, fn: float, freq_bound: 'Noise.FreqBound', hs: List[float],
                    noise_classes: List[Type['Noise']], verbose: bool=False):
        self.fn: float = fn
        self.freq_bound: 'Noise.FreqBound' = freq_bound
        self.hs: List[float] = hs
        self.noise_classes: List[Type['Noise']] = noise_classes
        self.verbose = verbose

class PhaseGenInst:
    """
    Parallel total phase generator instance.
    """

    def __init__(self, noise_gens: List['NoiseGenInst'], fn: Optional[float]=None):
        self._noise_gens = noise_gens
        self._time: List[float] = np.array([0]) # type: ignore
        self._nom_phase: List[float] = np.array([0]) # type: ignore
        self._fn: float
        if len(noise_gens) == 0:
            if fn is None:
                print('ERROR: no nominal frequency provided.')
                self._fn = 0
            else:
                self._fn = fn
        else:
            self._fn = self._noise_gens[0].noise.fn

    @classmethod
    def generate_0(cls: Type[PT], data: 'GenerateData') -> PT:
        """Factory method to construct a phase generator instance with the given
        parameters."""
        return cls.generate_1(data.fn, data.freq_bound, data.hs,
                              data.noise_classes, data.verbose)

    @classmethod
    def generate_1(cls: Type[PT], fn: float, freq_bound: 'Noise.FreqBound',
                 hs: List[float], noise_classes: List[Type['Noise']],
                 verbose: bool=False) -> PT:
        """Factory method to construct a phase generator instance with the given
        parameters."""
        noise_gens: List['NoiseGenInst'] = []
        for h, noise_cls in zip(hs, noise_classes):
            noise_source: 'Noise' = noise_cls.generate(fn, h, freq_bound)
            noise_gens.append(NoiseGenInst(noise_source, verbose))
        return cls(noise_gens)

    @property
    def phase(self) -> List[float]:
        """The total phase array, including nomial and excess phase."""
        exc_phase: List[float] = np.zeros(self._time.size) # type: ignore
        for noise_gen in self._noise_gens:
            exc_phase += noise_gen.phase
        return self._nom_phase + exc_phase

    @property
    def time(self) -> List[float]:
        """The time instances."""
        return self._time

    @property
    def fn(self) -> float:
        """The nominal oscillation frequency."""
        return self._fn

    def add_time_points(self, time_points: List[float]) -> None:
        """Run the add_time_points method for each of the internal noise
        generators."""
        self._time = np.block([self._time, np.array(time_points)]) # type: ignore
        self._time.sort() # type: ignore
        self._nom_phase: List[float] = 2 * np.pi * self._fn * self._time # type: ignore
        for noise_gen in self._noise_gens:
            noise_gen.add_time_points(time_points)

    def plot(self) -> None:
        """Plot this phase generator instance."""
        _, axs = plt.subplots(2) # type: ignore
        axs[0].set_title('Total phase') # type: ignore
        max_hline: int = int(np.ceil(max(self.phase) / 2 / np.pi))
        for index in range(1, max_hline):
            axs[0].axhline(y = 2 * np.pi * index, color = 'r', # type: ignore
                           linestyle = 'dashed')
        axs[0].plot(self._time, self.phase, 'o', color='black') # type: ignore
        axs[1].set_title('Noise phases') # type: ignore
        axs[1].plot(self._time, # type: ignore
                    self._nom_phase, label='Nominal phase')
        for noise_gen in self._noise_gens:
            axs[1].plot(self._time, noise_gen.phase, # type: ignore
                        label=f'{noise_gen.noise.title} phase')
        axs[1].legend() # type: ignore
        plt.show() # type:ignore

class NoiseGenInst:
    """
    Parallel noise generator instance.
    """

    def __init__(self, noise: 'Noise', verbose: bool=False) -> None:
        self._noise: Noise = noise
        self._time: List[float] = np.array([0]) # type: ignore
        self._phase: List[float] = np.array([0]) # type: ignore
        self._matrix_s: List[List[float]] = np.zeros((0,0)) # type: ignore
        self._plot_time: Optional[List[float]] = None
        self._plot_mu: Optional[List[float]] = None
        self._plot_s: Optional[List[List[float]]] = None
        self._plot_time_points: Optional[List[float]] = None
        self._plot_phase: Optional[List[float]] = None
        self._verbose = verbose

    @property
    def phase(self) -> List[float]:
        """The calculated noise source induced phase."""
        return self._phase

    @property
    def time(self) -> List[float]:
        """The time instances."""
        return self._time

    @property
    def noise(self) -> 'Noise':
        """This generator's noise source."""
        return self._noise

    def get_cond(self, time_points: List[float],
                  update_sigma: bool=False) \
            -> Tuple[List[float], List[List[float]]]:
        """Generate conditional mean and covariance matrices, given the points
        already in this generator. If update_sigma, the internal sigma matrix
        will be updated."""
        s11 = self._matrix_s
        s12 = self._noise.kernel(self._time[1:], time_points)
        solved: List[List[float]]
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                solved = scipy.linalg.solve(s11, s12, assume_a='pos').T # type: ignore
            except np.linalg.linalg.LinAlgError: # type: ignore
                if self._verbose:
                    print(f'{self._noise.title} generates a singular matrix, '
                          'using least squares now.')
                solved = np.linalg.lstsq(s11, s12, rcond=None)[0].T # type: ignore
            except Warning: # type: ignore
                if self._verbose:
                    print(f'{self._noise.title} generates ill-conditioned '
                          'matrix, using least squares now.')
                solved = np.linalg.lstsq(s11, s12, rcond=None)[0].T # type: ignore
        res: List[List[float]] = s11@solved.T-s12 # type: ignore
        norm_s12: float = np.linalg.norm(s12) # type: ignore
        norm: float
        if norm_s12 != 0:
            norm = np.linalg.norm(res) / norm_s12 # type: ignore
        else:
            norm = np.linalg.norm(res) # type: ignore
        # print(f'{self._noise.title}: det = {np.linalg.det(s11)}') # type: ignore
        if norm > 1e-3:
            raise Exception(f'Error norm too large: {norm}') # pylint: disable=broad-exception-raised
        s22 = self._noise.kernel(time_points)
        s2: List[List[float]] = s22 - (solved @ s12) # type: ignore
        m2: List[float] = solved @ self._phase[1:] # type: ignore
        if update_sigma:
            self._matrix_s = np.block([[s11, s12],[s12.transpose(), s22]]) # type: ignore
        return m2, s2 # type: ignore

    def add_time_points(self, time_points: List[float]) -> None:
        """Add the given time_points ot the internal time points. Calculate the
        phase for the new list of time points. Also Update the Sigma array. Sort
        the newly obtained time, phase and sigma arrays."""
        if len(self._time) == 0:
            mean_matrix: List[List[float]] = np.zeros(len(time_points)) # type: ignore
            self._matrix_s = self._noise.kernel(time_points)
            phases: List[float] = npr.multivariate_normal(mean_matrix, # type: ignore
                                                          self._matrix_s) # type: ignore
        else:
            m2, s2 = self.get_cond(time_points, update_sigma=True)
            phases: List[float] = npr.multivariate_normal(m2, s2) # type: ignore
        self._phase = cast(List[float], np.block([self._phase, phases])) # type: ignore
        self._time = cast(List[float], np.block([self._time, np.array(time_points)])) # type: ignore
        self._time, indexes, self._phase = [np.array(list(x)) # type: ignore
               for x in zip(*sorted(zip(self._time, # type: ignore
                                        range(len(self._time)), # type: ignore
                                        self._phase)))] # type: ignore
        indexes: List[int] = [i-1 for i in indexes if i != 0] # type: ignore
        self._matrix_s[:] = self._matrix_s[:, indexes][indexes, :] # type: ignore

    def generate_interval(self, time_interval: float, dt: Optional[float]=None,
                         verbose: bool=False) -> None:
        """
        Add interval points, until the dt is small enough.
        For the moment, this method is not in use.
        """
        nb_points: int
        if dt is not None:
            nb_points = int(np.ceil(time_interval / dt))
        else:
            nb_points = 1000
        self.add_time_points(np.array([time_interval])) # type: ignore
        if verbose:
            self.plot(show_update=True)
        intervals: List[Tuple[float, float]] = [(0, time_interval)]
        while len(self._time) < nb_points:
            new_intervals: List[Optional[Tuple[float, float]]] \
                = [None] * (2 * len(intervals))
            points: List[float] = np.zeros(len(intervals)) # type: ignore
            for index, interval in enumerate(intervals):
                mid = sum(interval) / 2
                points[index] = mid
                new_intervals[2 * index] = (interval[0], mid)
                new_intervals[2 * index + 1] = (mid, interval[1])
            self.add_time_points(points)
            intervals = cast(List[Tuple[float, float]], new_intervals)
            if verbose:
                self.plot(show_update=True)
        if verbose:
            self.plot()

    def plot(self, nb_points: int=1000, show_update: bool=False) -> None:
        """Plot this noise generator instance"""
        ran: float = max(self._time) * 1.1
        xs: List[float] = np.linspace(0, ran, nb_points).reshape(-1, 1) # type: ignore
        if show_update:
            if self._plot_time is None:
                self._plot_time = xs.reshape(1,-1) # type: ignore
                self._plot_mu, self._plot_s = self.get_cond(xs)
                self._plot_time_points = self._time.copy()
                self._plot_phase = self._phase.copy()
                return
        new_plot_time: List[float] = xs.reshape(1,-1) # type: ignore
        new_plot_mu, new_plot_s = self.get_cond(xs)
        new_plot_time_points = self._time.copy()
        new_plot_phase = self._phase.copy()
        _, axs = plt.subplots(1) # type: ignore
        s2: List[List[float]]
        if show_update:
            new_time: List[float] = []
            new_sample: List[float] = []
            assert self._plot_time_points is not None
            for ti, t in enumerate(self._time):
                if t not in self._plot_time_points:
                    new_time += [t]
                    new_sample += [self._phase[ti]]
            assert self._plot_time is not None
            assert self._plot_mu is not None
            assert self._plot_s is not None
            s2 = np.sqrt(np.diag(self._plot_s)) # type: ignore
            axs.fill_between(self._plot_time.flat, # type: ignore
                             self._plot_mu-2*s2, # type: ignore
                             self._plot_mu+2*s2, # type: ignore
                             color='red', alpha=0.15,
                             label='$2 \\sigma_{2|1}$') # type: ignore
            axs.plot(self._plot_time.flat, self._plot_mu.flat, # type: ignore
                     '-', color='red', label='$\\mu_{2|1}$') # type: ignore
            axs.plot(self._plot_time_points, self._plot_phase, # type: ignore
                     'o', color='black', label='old data')
            axs.plot(new_time, new_sample, 'o', color='orange', # type: ignore
                     label='new data')
        else:
            s2 = np.sqrt(np.diag(new_plot_s)) # type: ignore
            axs.fill_between(xs.flat, new_plot_mu - 2 * s2, # type: ignore
                             new_plot_mu + 2 * s2, # type: ignore
                             color='red', alpha=0.15,
                             label='$2 \\sigma_{2|1}$') # type: ignore
            axs.plot(xs, new_plot_mu, '-', color='red', # type: ignore
                     label='$\\mu_{2|1}$') # type: ignore
            axs.plot(self._time, self._phase, 'o', color='black', # type: ignore
                     label='data')
        axs.legend() # type: ignore
        self._plot_time = new_plot_time
        self._plot_mu = new_plot_mu
        self._plot_s = new_plot_s
        self._plot_time_points = new_plot_time_points
        self._plot_phase = new_plot_phase
        plt.show() # type: ignore
