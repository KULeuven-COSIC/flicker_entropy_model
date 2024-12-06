# Mathematical Model Python Scripts

This folder contains the Python scripts for simulating the mathematical model.

## Scripts

The following modules are available:
- **data_reader.py**: This module contains data storage functionality.
- **distribution.py**: This module contains empirical distribution functionality.
- **jitter_gen.py**: This module simulates a jitter measurement with a single RO. It Measures accumulated phase for nb_exp times at the given acc_times time points. The results are stored in the *simulation_data/jitter_gen/* folder.
- **noises.py**: This module contains the noise phase generation code. The results are stored in the *simulation_data/fn:[noise strength]/* folder.
- **monte_carlo_integrate.py**: This module contains the functionality to perform Monte Carlo integration, using the `Noise` class.