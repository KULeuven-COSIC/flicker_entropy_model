"""This module contains data storage functionality."""

import sys
from os import getcwd, listdir, makedirs
from os.path import join, isfile, isdir
import csv
from typing import Optional, List, Any, Dict, Tuple, Type
import numpy as np
sys.path.append(getcwd())
from math_model.python import distribution # pylint: disable=wrong-import-position

class DataManager:
    """A class containing data managing functionality."""

    _default_db_folder = join('math_model', 'simulation_data')

    def __init__(self, file_name: str, db_folder: Optional[str]=None):
        self._file_name: str = file_name
        self._db_folder: str
        if db_folder is None:
            self._db_folder = DataManager._default_db_folder
        else:
            self._db_folder = db_folder

    @classmethod
    def generate(cls: Type['DataManager'], db_folder: Optional[str]=None) \
        -> List['DataManager']:
        """Generate a list of data managers for all files in the given db_folder."""
        if db_folder is None:
            db_folder = DataManager._default_db_folder
        result: List['DataManager'] = []
        files = [f for f in listdir(db_folder) if isfile(join(db_folder, f))]
        for file_ in files:
            result.append(DataManager(file_, db_folder))
        return result

    @property
    def file(self) -> str:
        """The complete file path that this data manager is writing to/reading from."""
        return join(self._db_folder, self._file_name)

    @property
    def file_name(self) -> str:
        """This data manager's file name."""
        return self._file_name

    def _file_exists(self) -> bool:
        return isfile(self.file)

    def init_file(self, col_headers: List[str], over_write: bool=False,
                  verbose: bool=True) -> None:
        """Initialize the file with the given column headers. If the file already exists,
        only overwrite the file if over_write is set to True."""
        if self._file_exists():
            if verbose:
                print(f'Warning, the file {self.file} already exists.')
            if not over_write:
                return
        with open(self.file, 'w', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(col_headers)

    def append_row(self, row_data: List[Any]) -> None:
        """Append the given data row to this manager's file, if the file exists."""
        if not self._file_exists():
            print(f'Error, file {self.file} does not exist.')
            return
        with open(self.file, 'a', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(row_data)

    def append_rows(self, rows_data: List[List[Any]]) -> None:
        """Append the given data rows to this manager's file, if the files exists."""
        if not self._file_exists():
            print(f'Error, file {self.file} does not exist.')
            return
        with open(self.file, 'a', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for row in rows_data:
                csv_writer.writerow(row)

    @staticmethod
    def _parse_entry(entry: str) -> Any:
        if entry.lower() == 'true':
            return True
        if entry.lower() == 'false':
            return False
        try:
            return int(entry, base=10)
        except ValueError:
            try:
                return float(entry)
            except ValueError:
                return entry

    def get_header(self, verbose: bool=True) -> Optional[List[Any]]:
        """Get the parsed header from this manager's file. If the file does not exist,
        None is returned"""
        if not self._file_exists():
            if verbose:
                print(f'Error, file {self.file} does not exist.')
            return None
        result: List[Any] = []
        with open(self.file, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for entry in next(csv_reader):
                result.append(DataManager._parse_entry(entry))
        return result

    def get_data(self, verbose: bool=True) -> Optional[List[List[Any]]]:
        """Get the parsed data from this manager's file. If the file does not exist,
        None is returned."""
        if not self._file_exists():
            if verbose:
                print(f'Error, file {self.file} does not exist.')
            return None
        result: List[List[Any]] = []
        with open(self.file, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            for row in csv_reader:
                result_row: List[Any] = []
                for entry in row:
                    result_row.append(DataManager._parse_entry(entry))
                result.append(result_row)
        return result

class DistributionManager:
    """Class for managing the storage of distributions generated during a WF sweep."""

    _nb_files_per_folder = 128

    def __init__(self, base_db_folder: str='simulation_data') -> None:
        self._db_folder = base_db_folder

    def _get_folder_name(self,
                         fn: float,
                         h_flicker: Optional[float]=None,
                         sample_per: Optional[float]=None,
                         hist_type: Optional[str]=None) -> str:
        result = join(self._db_folder, DistributionManager._fn_folder(fn))
        if h_flicker is None:
            return result
        result = join(result, DistributionManager._flicker_folder(h_flicker))
        if sample_per is None:
            return result
        result = join(result, DistributionManager._sample_per_folder(sample_per))
        if hist_type is None:
            return result
        result = join(result, DistributionManager._hist_type_folder(hist_type))
        return result

    @staticmethod
    def _get_sub_folder_name_static(bit_nb: int,
                                    hist_depth: Optional[int]=None,
                                    hist_value: Optional[int]=None) -> str:
        result = DistributionManager._bit_nb_folder(bit_nb)
        if hist_depth is None:
            return result
        result = join(result, DistributionManager._depth_folder(hist_depth))
        if hist_value is None:
            return result
        range_start = int(hist_value / DistributionManager._nb_files_per_folder) \
            * DistributionManager._nb_files_per_folder
        range_stop = range_start + DistributionManager._nb_files_per_folder
        result = join(result, DistributionManager._hist_value_folder(range_start,
                                                                             range_stop))
        return result

    @staticmethod
    def _get_sub_folder_name_id(id_: str) -> str:
        bit_nb, hist_bit_nbs, hist_value, _ = DistributionManager.parse_dist_id(id_)
        return DistributionManager._get_sub_folder_name_static(bit_nb, len(hist_bit_nbs),
                                                               hist_value)

    @staticmethod
    def _get_sub_folder_name_dist(dist: 'distribution.Distribution') -> str:
        return DistributionManager._get_sub_folder_name_id(dist.id_)

    @staticmethod
    def _fn_folder(fn: float) -> str:
        return f'fn:{fn:9.3e}'

    @staticmethod
    def _parse_fn(folder_name: str) -> float:
        return float(folder_name.split(':')[-1].split('/')[0])

    @staticmethod
    def _flicker_folder(h_flicker: float) -> str:
        return f'flicker:{h_flicker:9.3e}'

    @staticmethod
    def _parse_h_flicker(folder_name: str) -> float:
        return float(folder_name.split(':')[-1].split('/')[0])

    @staticmethod
    def _sample_per_folder(sample_per: float) -> str:
        return f'sample_per:{sample_per:9.3e}'

    @staticmethod
    def _parse_sample_per(folder_name: str) -> float:
        return float(folder_name.split(':')[-1].split('/')[0])

    @staticmethod
    def _hist_type_folder(hist_type: str) -> str:
        return f'hist_type:{hist_type}'

    @staticmethod
    def _parse_hist_type(folder_name: str) -> str:
        return folder_name.split(':')[-1].split('/')[0]

    @staticmethod
    def _bit_nb_folder(bit_nb: int) -> str:
        return f'bit_nb:{bit_nb}'

    @staticmethod
    def _parse_bit_nb(folder_name: str) -> int:
        return int(folder_name.split(':')[-1].split('/')[0])

    @staticmethod
    def _depth_folder(depth: int) -> str:
        return f'depth:{depth}'

    @staticmethod
    def _parse_depth(folder_name: str) -> int:
        return int(folder_name.split(':')[-1].split('/')[0])

    @staticmethod
    def _hist_value_folder(start: int, stop: int) -> str:
        return f'hist_value:{start}>{stop}'

    @staticmethod
    def _parse_hist_value(folder_name: str) -> Tuple[int, int]:
        hist_part = folder_name.split(':')[-1]
        splitted = hist_part.split('>')
        return (int(splitted[0]), int(splitted[-1].split('/')[0]))

    def store_dist(self, dist: 'distribution.Distribution',
                   fn: float, h_flicker: float, sample_per: float, hist_type: str) -> None:
        """Store the given distribution in the folder structure."""
        folder_name_base = self._get_folder_name(fn, h_flicker, sample_per, hist_type)
        sub_folder_name = DistributionManager._get_sub_folder_name_dist(dist)
        folder_name = join(folder_name_base, sub_folder_name)
        if not isdir(folder_name):
            makedirs(folder_name)
        dist.store_file(folder_name)

    def get_dist(self, id_: str, nb_bins: int,
                 fn: float, h_flicker: float, sample_per: float, hist_type: str) \
        -> 'distribution.Distribution':
        """Get the stored distribution. Return a new distribution
        if the distribution does not exist."""
        folder_name_base = self._get_folder_name(fn, h_flicker, sample_per, hist_type)
        sub_folder_name = DistributionManager._get_sub_folder_name_id(id_)
        folder_name = join(folder_name_base, sub_folder_name)
        result = distribution.Distribution.from_file(id_, nb_bins, folder_name,
                                                     verbose=False)
        if result is None:
            bit_nb, _, _, _ = DistributionManager.parse_dist_id(id_)
            nom_phase = 2 * np.pi * fn * bit_nb * sample_per
            return distribution.Distribution(id_, nb_bins, nom_phase)
        return result

    def get_all_hist_value_range(self, fn: float, h_flicker: float, sample_per: float,
                                 hist_type: str, bit_nb: int, hist_depth: int,
                                 range_start: int) \
        -> List[Tuple[str, int]]:
        """Get all distributions inside the given hist_value folder."""
        folder_name_base = self._get_folder_name(fn, h_flicker, sample_per, hist_type)
        sub_folder_name = DistributionManager._get_sub_folder_name_static(bit_nb,
                                                                        hist_depth,
                                                                        range_start)
        folder_name = join(folder_name_base, sub_folder_name)
        all_files = [f for f in listdir(folder_name)
                     if isfile(join(folder_name, f))]
        return [distribution.Distribution.parse_file_name(f) for f in all_files]

    def get_all_hist_depth(self, fn: float, h_flicker: float, sample_per: float,
                           hist_type: str, bit_nb: int, hist_depth: int) \
        -> Dict[Tuple[int, int], List[Tuple[str, int]]]:
        """Get all distributions inside the given hist_depth folder."""
        folder_name_base = self._get_folder_name(fn, h_flicker, sample_per, hist_type)
        sub_folder_name = DistributionManager._get_sub_folder_name_static(bit_nb,
                                                                        hist_depth)
        folder_name = join(folder_name_base, sub_folder_name)
        all_folders = [f for f in listdir(folder_name)
                       if isdir(join(folder_name, f))]
        result: Dict[Tuple[int, int], List[Tuple[str, int]]] = {}
        for f in all_folders:
            (start, stop) = DistributionManager._parse_hist_value(f)
            result[(start, stop)] = self.get_all_hist_value_range(fn, h_flicker, sample_per,
                                                                  hist_type, bit_nb,
                                                                  hist_depth, start)
        return result

    def get_all_bit_nb(self, fn: float, h_flicker: float, sample_per: float,
                       hist_type: str, bit_nb: int) \
        -> Dict[int, Dict[Tuple[int, int], List[Tuple[str, int]]]]:
        """Get all distributions inside the given bit_nb folder."""
        folder_name_base = self._get_folder_name(fn, h_flicker, sample_per, hist_type)
        sub_folder_name = DistributionManager._get_sub_folder_name_static(bit_nb)
        folder_name = join(folder_name_base, sub_folder_name)
        all_folders = [f for f in listdir(folder_name)
                       if isdir(join(folder_name, f))]
        result: Dict[int, Dict[Tuple[int, int], List[Tuple[str, int]]]] = {}
        for f in all_folders:
            depth = DistributionManager._parse_depth(f)
            result[depth] = self.get_all_hist_depth(fn, h_flicker, sample_per, hist_type,
                                                    bit_nb, depth)
        return result

    def get_all_dists(self, fn: float, h_flicker: float, sample_per: float, hist_type: str) \
        -> List[Tuple[str, int]]:
        """Get all distributions with the given h_flicker, sample_per and hist_type."""
        folder_name = self._get_folder_name(fn, h_flicker, sample_per, hist_type)
        all_folders = [f for f in listdir(folder_name)
                       if isdir(join(folder_name, f))]
        result: List[Tuple[str, int]] = []
        for f in all_folders:
            bit_nb = DistributionManager._parse_bit_nb(f)
            all_bit_nb = self.get_all_bit_nb(fn, h_flicker, sample_per, hist_type, bit_nb)
            result += [c for a in all_bit_nb.values() for b in a.values() for c in b]
        return result

    def get_all_hist_types(self, fn: float, h_flicker: float, sample_per: float) \
        -> Dict[str, List[Tuple[str, int]]]:
        """Get all distributions with the given h_flicker and sample_per."""
        folder_name = self._get_folder_name(fn, h_flicker, sample_per)
        all_folders = [f for f in listdir(folder_name)
                       if isdir(join(folder_name, f))]
        result: Dict[str, List[Tuple[str, int]]] = {}
        for f in all_folders:
            hist_type = DistributionManager._parse_hist_type(f)
            result[hist_type] = self.get_all_dists(fn, h_flicker, sample_per, hist_type)
        return result

    def get_all_sample_pers(self, fn: float, h_flicker: float) \
        -> Dict[float, Dict[str, List[Tuple[str, int]]]]:
        """Get all distributions with the given h_flicker."""
        folder_name = self._get_folder_name(fn, h_flicker)
        all_folders = [f for f in listdir(folder_name)
                       if isdir(join(folder_name, f))]
        result: Dict[float, Dict[str, List[Tuple[str, int]]]] = {}
        for f in all_folders:
            sample_per = DistributionManager._parse_sample_per(f)
            result[sample_per] = self.get_all_hist_types(fn, h_flicker, sample_per)
        return result

    def get_all_h_flicker(self, fn: float) \
        -> Dict[float, Dict[float, Dict[str, List[Tuple[str, int]]]]]:
        """Get a dictionary containing all data stored for the oscillation frequency."""
        folder_name = self._get_folder_name(fn)
        all_folders = [f for f in listdir(folder_name)
                       if isdir(join(folder_name, f))]
        result: Dict[float, Dict[float, Dict[str, List[Tuple[str, int]]]]] = {}
        for f in all_folders:
            h_flicker = DistributionManager._parse_h_flicker(f)
            result[h_flicker] = self.get_all_sample_pers(fn, h_flicker)
        return result

    def get_all_fn(self) -> Dict[float, Dict[float, Dict[float, Dict[str, List[Tuple[str, int]]]]]]:
        """Get a dictionary containing all data stored at the base directory."""
        folder_name = self._db_folder
        all_folders = [f for f in listdir(folder_name)
                       if isdir(join(folder_name, f))]
        result: Dict[float, Dict[float, Dict[float, Dict[str, List[Tuple[str, int]]]]]] = {}
        for f in all_folders:
            fn = DistributionManager._parse_fn(f)
            result[fn] = self.get_all_h_flicker(fn)
        return result

    def get_all(self) \
        -> Dict[float, Dict[float, Dict[float, Dict[str, List['distribution.Distribution']]]]]:
        """Get all stored distributions."""
        all_dists = self.get_all_fn()
        result: Dict[float, Dict[float, Dict[float,
                                             Dict[str, List['distribution.Distribution']]]]] \
            = {}
        for fn, fn_dists in all_dists.items():
            fn_result: Dict[float, Dict[float, Dict[str, List['distribution.Distribution']]]] = {}
            for h_flicker, h_flicker_dists in fn_dists.items():
                h_flicker_result: Dict[float, Dict[str, List['distribution.Distribution']]] = {}
                for sample_per, sample_per_dists in h_flicker_dists.items():
                    sample_per_result: Dict[str, List['distribution.Distribution']] = {}
                    for hist_type, hist_type_dists in sample_per_dists.items():
                        sample_per_result[hist_type] \
                            = [self.get_dist(*d, fn, h_flicker, sample_per, hist_type)
                            for d in hist_type_dists]
                    h_flicker_result[sample_per] = sample_per_result
                fn_result[h_flicker] = h_flicker_result
            result[fn] = fn_result
        return result

    @staticmethod
    def get_dist_id(bit_nb: int, hist_bit_nbs: List[int], hist_value: int, white: bool) -> str:
        """Construct the distribution id string."""
        if not hist_bit_nbs:
            return f'{bit_nb}|||{"w" if white else "f"}'
        if len(hist_bit_nbs) == 1:
            return f'{bit_nb}|{hist_bit_nbs[0]}|{hist_value}|{"w" if white else "f"}'
        return (f'{bit_nb}|{"#".join([str(b) for b in hist_bit_nbs])}'
                f'|{hist_value}|{"w" if white else "f"}')

    @staticmethod
    def parse_dist_id(id_: str) -> Tuple[int, List[int], int, bool]:
        """Parse the data from a distribution id string."""
        parts = id_.split('|')
        bit_nb = int(parts[0])
        hist_bit_nbs = [int(p) for p in parts[1].split('#') if p]
        hist_value = int(parts[2]) if parts[2] else 0
        white = parts[3] == 'w'
        return (bit_nb, hist_bit_nbs, hist_value, white)

    @staticmethod
    def get_hist_value_start_stop(hist_value: int) -> Tuple[int, int]:
        """Get the range where the given hist_value falls into."""
        start = int(hist_value / DistributionManager._nb_files_per_folder) \
                * DistributionManager._nb_files_per_folder
        return (start, start + DistributionManager._nb_files_per_folder)
