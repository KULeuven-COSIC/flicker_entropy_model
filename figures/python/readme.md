# Figure Generation Python Scripts

This folder contains the figure generation Python scripts.

## Scripts

The scripts are divided in the following categories.

### Excess Phase Process

- **gauss_process.py**: Generate a figure showing the Gaussian process in action for both white FM and flicker FM noise.

### ERO TRNG Entropy Model

- **gauss_one.py**: Generate a figure showing a Gaussian PDF, with an integration area under the curve to visualize the probability of sampling a one.
- **two_sample.py**: Generate a figure showing a multivariate Gaussian PDF, for two samples.
- **two_diff.py**: Generate a figure showing a multivariate Gaussian PDF, for two samples. Highlight how the PDF changes when the first sample turns out to be equal to one.
- **sample_two.py**: Generate a figure showing the excess phase PDF for a second sample, when the first sample bit is known to equal one. Also show an integration area under the curve to visualize the probability of sampling a second one.

### Worst-case Entropy

- **sample_two_worst.py**: Generate a figure showing the excess phase PDF for a second sample, when the first sample bit is known to equal one. Also show an integration area under the curve for the worst case entropy, to visualize the probability of sampling a second one.
- **worst_ent_acc_time.py**: Generate a figure showing the relation worst entropy vs. accumulation time. For different white noise strengths.

### Model Simulation

- **noise_strength.py**: Generate a noise strength plot, comparing the three different flicker noise strengths used and showing the three different white-flicker noise corners.

#### Known Phase

- **known_phase_0.py**: Generate figure showing standard deviation and worst-case entropy for a single bit, given the knowledge of n (0 - 10) previous phase values.
- **known_phase_1.py**: Generate figure showing standard deviation and worst-case entropy for a single bit, given the knowledge of n (0 - 10) previous phase values.
- **known_phase_2.py**: Generate figure showing standard deviation and worst-case entropy for a single bit, given the knowledge of n (0 - 10) previous phase values.
- **flicker_acc_time.py**: Generate a figure showing the worst-case entropy, given the knowledge of the previous phase for flicker noise versus the accumulation time between the bits. Show curves for multiple flicker noise magnitudes.

#### Known Output

- **comb_bit_entropy_0.py**: Generate a figure showing the worst-case Shannon entropy, when up to b previous sample bits are known.
- **comb_bit_entropy_1.py**: Generate a figure showing the worst-case Shannon entropy, when up to b previous sample bits are known.
- **comb_bit_entropy_2.py**: Generate a figure showing the worst-case Shannon entropy, when up to b previous sample bits are known.

## Script Options

The following script arguments are available:
- `-v`: Enable verbose output.
- `-d`: Generate processed data and store in the *data/* folder.
- `-q`: Quit the script as soon as the processed data is generated, without generating the figure. Should only be used in combination with the `-d` argument.
- `-l`: For lengthy execution times, the log option might be available, indicating the time required to execute the script.