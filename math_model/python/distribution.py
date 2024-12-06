"""A module containing empirical distribution functionality."""

import sys
from os import getcwd
from typing import List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt # type: ignore
import numpy as np
sys.path.append(getcwd())
from math_model.python import data_reader # pylint: disable=wrong-import-position

class Distribution:
    """A class representing a distribution created from emperical samples."""

    _range_mult: int = 2
    _init_nb_shifts = 100
    _nb_samples_one_shift = 1000

    def __init__(self, id_: str, nb_bins: int, nominal_phase: float):
        self._nb_bins = nb_bins
        self._id = id_
        self._shift = nominal_phase % np.pi
        self._bin_amounts: List[List[int]] = [[]]
        self._bin_shifts: List[float] = [0]
        self._bin_width = 0 # As a factor of pi
        self._is_init = False
        self._nb_smaller: int = 0
        self._nb_larger: int = 0
        self._bin_min = 0
        self._bin_max = 0
        self._bin_merge = False
        self._nb_merge = 1
        self._real_nb_bins = 0

    @property
    def id_(self) -> str:
        """This distribution's id value."""
        return self._id

    @property
    def nb_bins(self) -> int:
        """This distribution's number of bins, as a target. This is
        not the real number of bins."""
        return self._nb_bins

    @property
    def _bin_starts(self) -> List[float]:
        if self._bin_merge:
            return [(self._bin_min + 2 * self._nb_merge * bin_) * np.pi - self._shift
                    for bin_ in range(int(self._real_nb_bins / 2))]
        return [(self._bin_min + self._bin_width * bin_) * np.pi - self._shift
                for bin_ in range(self._real_nb_bins)]

    @property
    def _bin_ends(self) -> List[float]:
        if self._bin_merge:
            return [(self._bin_min + 2 * self._nb_merge * (bin_ + 1)) * np.pi - self._shift
                    for bin_ in range(int(self._real_nb_bins / 2))]
        return [(self._bin_min + self._bin_width * (bin_ + 1)) * np.pi - self._shift
                for bin_ in range(self._real_nb_bins)]

    @property
    def bin_mids(self) -> List[float]:
        """Get the bin midpoints for this distribution."""
        if self._bin_merge:
            return [(self._bin_min + (2 * self._nb_merge - 1) / 2 + (bin_ % 2) \
                    + int(bin_ / 2) * 2 * self._nb_merge) * np.pi \
                    - self._shift
                    for bin_ in range(self._real_nb_bins)]
        return [(self._bin_min + (bin_ + 0.5) * self._bin_width) * np.pi - self._shift
                for bin_ in range(self._real_nb_bins)]

    @property
    def nb_samples(self) -> int:
        """The total number of samples received."""
        return sum(self._bin_amounts[0]) + self._nb_smaller + self._nb_larger

    def _init_dist(self, samples: List[float]) -> None:
        self._is_init = True
        nb_added = len(samples)
        if nb_added:
            sample_min = min(samples)
            sample_max = max(samples)
        else:
            sample_min = 0
            sample_max = 0
        sample_range = sample_max - sample_min
        if nb_added < 10:
            ext_sample_range = max(20 * np.pi, sample_range * Distribution._range_mult * 2)
        else:
            ext_sample_range = sample_range * Distribution._range_mult
        range_added = ext_sample_range - sample_range
        self._bin_min = int(np.floor((sample_min - range_added / 2 + self._shift) / np.pi))
        self._bin_max = int(np.ceil((sample_max + range_added / 2 + self._shift) / np.pi))
        nb_pi_blocks = self._bin_max - self._bin_min
        if nb_pi_blocks >= 2 * self._nb_bins:
            self._bin_merge = True
            self._nb_merge = int(np.floor(nb_pi_blocks / self._nb_bins))
            self._bin_width = self._nb_merge
            self._real_nb_bins = int(np.ceil(nb_pi_blocks / self._nb_merge / 2) * 2)
            added_nb_bins = self._real_nb_bins * self._nb_merge - nb_pi_blocks
            self._bin_min -= int(np.floor(added_nb_bins / 2))
            self._bin_max += int(np.ceil(added_nb_bins / 2))
        else:
            nb_bins_per_pi = int(np.ceil(self._nb_bins / nb_pi_blocks))
            self._bin_width = 1 / nb_bins_per_pi
            self._real_nb_bins = nb_bins_per_pi * nb_pi_blocks
        self._bin_amounts = [[0] * self._real_nb_bins for _ in range(Distribution._init_nb_shifts)]
        self._bin_shifts = [sh / Distribution._init_nb_shifts
                            for sh in range(Distribution._init_nb_shifts)]

    def add_samples(self, samples: List[float]) -> None:
        """Provide new samples to this distribution. If samples contains less
        than 5 samples, the distribution is initialized in the range -10 pi rad
        to 10 pi rad."""
        if not self._is_init:
            self._init_dist(samples)
        if self._bin_merge:
            for amounts, shift in zip(self._bin_amounts, self._bin_shifts):
                for sample in samples:
                    mod_sample = (sample + self._shift) / np.pi + shift
                    if mod_sample < self._bin_min:
                        if not shift:
                            self._nb_smaller += 1
                        continue
                    if mod_sample >= self._bin_max:
                        if not shift:
                            self._nb_larger += 1
                        continue
                    block_nb = int((mod_sample - self._bin_min) / (2 * self._nb_merge))
                    amounts[block_nb * 2 + int(mod_sample - self._bin_min) % 2] += 1
        else:
            for amounts, shift in zip(self._bin_amounts, self._bin_shifts):
                for sample in samples:
                    mod_sample = (sample + self._shift) / np.pi + shift
                    if mod_sample < self._bin_min:
                        if not shift:
                            self._nb_smaller += 1
                        continue
                    if mod_sample >= self._bin_max:
                        if not shift:
                            self._nb_larger += 1
                        continue
                    try:
                        amounts[int((mod_sample - self._bin_min) / self._bin_width)] += 1
                    except IndexError as exc:
                        print('index check')
                        print(f'bin_min: {self._bin_min}')
                        print(f'bin_max: {self._bin_max}')
                        print(f'len(amounts): {len(amounts)}')
                        print(f'bin_width: {self._bin_width}')
                        print(f'mod_sample: {mod_sample}')
                        print(f'index: {int((mod_sample - self._bin_min) / self._bin_width)}')
                        raise IndexError('check the data!') from exc
        if len(self._bin_shifts) > 2:
            nb_samples = self.nb_samples
            if nb_samples:
                nb_shifts_keep = max(1, int(Distribution._init_nb_shifts \
                                            - np.ceil(Distribution._init_nb_shifts \
                                                      / Distribution._nb_samples_one_shift \
                                                      * nb_samples)))
                ents = self._entropies()
                _, self._bin_shifts[1:], self._bin_amounts[1:] \
                    = zip(*sorted(zip(ents[1:], self._bin_shifts[1:], self._bin_amounts[1:])))
                self._bin_shifts = self._bin_shifts[:nb_shifts_keep + 1]
                self._bin_amounts = self._bin_amounts[:nb_shifts_keep + 1]

    def _entropies(self) -> List[float]:
        nb_samples = sum(self._bin_amounts[0])
        if self._bin_merge:
            ps = (sum(amounts[::2]) / nb_samples for amounts in self._bin_amounts)
            return [-np.log2(max(p, 1 - p)) for p in ps]
        nb_bin_per_pi = int(1 / self._bin_width)
        nb_pi_blocks = int(self._real_nb_bins / nb_bin_per_pi)
        result = [0] * len(self._bin_shifts)
        for index, amounts in enumerate(self._bin_amounts):
            for block in range(0, nb_pi_blocks, 2):
                result[index] += sum(amounts[block * nb_bin_per_pi:(block + 1) * nb_bin_per_pi])
        return [-np.log2(max(p / nb_samples, 1 - p / nb_samples)) for p in result]

    @property
    def norm_hist(self) -> List[float]:
        """The normalized histogram derived from the provided samples."""
        nb_samples = sum(self._bin_amounts[0])
        if not nb_samples:
            return list(self._bin_amounts[0])
        return [p / nb_samples for p in self._bin_amounts[0]]

    @property
    def pdf(self) -> List[float]:
        """The probability distribution function derived from the provided samples."""
        return [n / (self._bin_width * np.pi) for n in self.norm_hist]

    @property
    def cdf(self) -> List[float]:
        """The cumulative distribution function derived from the provided samples."""
        n_hist = self.norm_hist
        return [sum(n_hist[:i + 1]) for i, _ in enumerate(n_hist)]

    @property
    def min_entropy(self) -> float:
        """"This distribution's min entropy."""
        if not self._is_init:
            return 0
        nb_samples = sum(self._bin_amounts[0])
        if self._bin_merge:
            p = sum(self._bin_amounts[0][::2]) / nb_samples
            return -np.log2(max(p, 1 - p))
        nb_bin_per_pi = int(1 / self._bin_width)
        nb_pi_blocks = int(self._real_nb_bins / nb_bin_per_pi)
        result = 0
        for block in range(0, nb_pi_blocks, 2):
            result += sum(self._bin_amounts[0][block * nb_bin_per_pi:(block + 1) \
                                                * nb_bin_per_pi])
        return -np.log2(max(result / nb_samples, 1 - result / nb_samples))

    @property
    def worst_min_entropy(self) -> Tuple[float, float]:
        """This distribution's worst min entropy."""
        if not self._is_init:
            return (0, 0)
        h0 = self.min_entropy
        if self._bin_merge:
            p1 = sum(self._bin_amounts[1][::2]) / sum(self._bin_amounts[1])
            h1 = -np.log2(max(p1, 1 - p1))
        else:
            nb_samples = sum(self._bin_amounts[1])
            nb_bin_per_pi = int(1 / self._bin_width)
            nb_pi_blocks = int(self._real_nb_bins / nb_bin_per_pi)
            p1 = 0
            for block in range(0, nb_pi_blocks, 2):
                p1 += sum(self._bin_amounts[1][block * nb_bin_per_pi:(block + 1) * nb_bin_per_pi])
            h1 = -np.log2(max(p1 / nb_samples, 1 - p1 / nb_samples))
        return (h1, self._bin_shifts[1]) if h1 < h0 else (h0, 0)

    def plot_hist(self) -> None:
        """Plot this distribution's accumulated histogram."""
        _, ax = plt.subplots(1,1) # type: ignore
        ax.plot(self.bin_mids, self.pdf, label='no shift') # type: ignore
        pdf_shifted = [a / sum(self._bin_amounts[1]) / self._bin_width / np.pi
                       for a in self._bin_amounts[1]]
        ax.plot(self.bin_mids, pdf_shifted, # type: ignore
                label=f'shift {self._bin_shifts[1]:9.3e}')
        y_max = max(*self.pdf, *pdf_shifted)
        pis = [i * np.pi - self._shift for i in range(self._bin_min, self._bin_max, 2)]
        for index, p in enumerate(pis):
            kwargs = {
                'color': 'grey',
                'alpha': 0.2
            }
            if not index:
                kwargs['label'] = 'pi mult'
            ax.fill_betweenx([0, y_max], [p, p], [p + np.pi, p + np.pi], # type: ignore
                            **kwargs) # type: ignore
        ax.set_title(f'Distribution histogram with {self.nb_samples} samples') # type: ignore
        ax.set_xlabel('Excess phase [rad]') # type: ignore
        ax.set_ylabel('Probability density [-]') # type: ignore
        ax.legend() # type: ignore
        plt.show() # type: ignore

    @property
    def file_name(self) -> str:
        """This distribution's file name."""
        return Distribution._construct_file_name(self._id, self._nb_bins)

    def store_file(self, db_folder: str='simulation_data') -> None:
        """Store this distribution into a file in the given db_folder."""
        data_manager = data_reader.DataManager(self.file_name, db_folder)
        nb_samples = self.nb_samples
        if not nb_samples:
            shifts_str = ''
        else:
            shifts_str = '#'.join(f'{s:5.3f}' for s in self._bin_shifts)
        meta_data = [f'{self._shift:9.7f}', shifts_str, f'{self._bin_width:9.3e}',
                     f'{self._bin_min:09d}', f'{self._bin_max:09d}',
                     f'{self._bin_merge:1d}', f'{self._nb_merge:09d}',
                     f'{self._real_nb_bins:09d}', f'{self._nb_smaller:09d}',
                     f'{self._nb_larger:09d}']
        data_manager.init_file(meta_data, over_write=True, verbose=False)
        if nb_samples:
            store_amounts: List[List[int]] = []
            for amounts in self._bin_amounts:
                amounts_store: List[int] = []
                zero_start = 0
                for index, a in enumerate(amounts):
                    if a:
                        if index != zero_start:
                            amounts_store.append(-(index  - zero_start))
                        zero_start = index + 1
                        amounts_store.append(a)
                    if index == self._real_nb_bins - 1:
                        if not a:
                            amounts_store.append(-(index + 1 - zero_start))
                store_amounts.append(amounts_store)
            data_manager.append_rows(store_amounts)

    @staticmethod
    def _construct_file_name(id_: str, nb_bins: int) -> str:
        return f'dist_id:{id_}_bins:{nb_bins}.csv'

    @staticmethod
    def _parse_meta_data(meta_data: List[Any]) -> Tuple[float, List[float], Union[float, int],
                                                        int, int, bool, int, int, int, int]:
        shift = float(meta_data[0])
        shifts_parts = meta_data[1].split('#')
        bin_shifts = [float(p) for p in shifts_parts if p]
        bin_min = int(meta_data[3])
        bin_max = int(meta_data[4])
        bin_merge = bool(meta_data[5])
        bin_width: Union[float, int]
        if bin_merge:
            bin_width = int(meta_data[2])
        else:
            bin_width = float(meta_data[2])
        nb_merge = int(meta_data[6])
        real_nb_bins = int(meta_data[7])
        nb_smaller = int(meta_data[8])
        nb_larger = int(meta_data[9])
        return (shift, bin_shifts, bin_width, bin_min, bin_max, bin_merge, nb_merge,
                real_nb_bins, nb_smaller, nb_larger)

    @classmethod
    def from_file(cls, id_: str, nb_bins: int, db_folder: str='simulation_data',
                  verbose: bool=True) -> Optional['Distribution']:
        """Parse a distribution object from a stored file. Return None if the
        distribution does not exist."""
        file_name = Distribution._construct_file_name(id_, nb_bins)
        data_manager = data_reader.DataManager(file_name, db_folder)
        meta_data = data_manager.get_header(verbose)
        if meta_data is None:
            if verbose:
                print(f'File: {data_manager.file} does not exist!')
            return None
        (shift, bin_shifts, bin_width, bin_min, bin_max, bin_merge, nb_merge, real_nb_bins,
         nb_smaller, nb_larger) = Distribution._parse_meta_data(meta_data)
        bin_amounts: List[List[int]]
        if bin_shifts:
            bin_amounts_temp = data_manager.get_data(verbose)
            if bin_amounts_temp is None:
                if verbose:
                    print(f'File: {data_manager.file} does not exist!')
                return None
            bin_amounts = []
            for amounts in bin_amounts_temp:
                ba: List[int] = []
                for a in amounts:
                    if a < 0:
                        ba += [0] * (-a)
                    else:
                        ba.append(a)
                bin_amounts.append(ba)
            init = True
        else:
            bin_amounts = [[]]
            bin_shifts = [0.0]
            init = False
        result = Distribution(id_, nb_bins, shift)
        result._bin_amounts = bin_amounts
        result._nb_smaller = nb_smaller
        result._nb_larger = nb_larger
        result._bin_shifts = bin_shifts
        if not bin_merge:
            if real_nb_bins:
                result._bin_width = (bin_max - bin_min) / real_nb_bins
            else:
                result._bin_width = bin_width
        else:
            result._bin_width = bin_width
        result._bin_min = bin_min
        result._bin_max = bin_max
        result._bin_merge = bin_merge
        result._nb_merge = nb_merge
        result._real_nb_bins = real_nb_bins
        result._is_init = init
        return result

    @staticmethod
    def parse_file_name(file_name: str) -> Tuple[str, int]:
        """Return the id and number of bins parsed from the given file name."""
        parts = file_name.split('_')
        id_ = str(parts[1].split(':')[-1])
        nb_bins = int(parts[2].split('.')[0].split(':')[-1])
        return (id_, nb_bins)

    def __str__(self) -> str:
        return f'Distribution nb: {self._nb_bins:09d}, id: {self._id}'
