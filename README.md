# The Hypyxic package

Hypyxic is a python code to compute sleep apnea specific hypoxic burden (SASHB) from oxygen desaturation signal and sleep data.

## About
This package was based on Matlab implementation of Hypoxic Burden made by Philip de Chazal at https://github.com/pdechazal/Hypoxic-Burden. 

This code was written by Matheus Araujo, more info at: https://www.matheusaraujo.com/

## References

**Kate Sutherland, Nadi Sadr, Yu Sun Bin, Kristina Cook, Hasthi U. Dissanayake, Peter A. Cistulli and Philip de Chazal**
"Comparative associations of oximetry patterns in Obstructive Sleep Apnea with incident cardiovascular disease", Sleep, 2022, DOI: 10.1093/sleep/zsac179

**Azarbarzin, Ali, Scott A. Sands, Katie L. Stone, Luigi Taranto-Montemurro, Ludovico Messineo, Philip I. Terrill, Sonia Ancoli-Israel et al.**
"The hypoxic burden of sleep apnoea predicts cardiovascular disease-related mortality: the Osteoporotic Fractures in Men Study and the Sleep Heart Health Study." European heart journal 40, no. 14 (2019): 1149-1157.

## Installation

1. Download the file hypyxic.py to the same folder that you run you Python script

## Test

```bash
python test_hypyxic.py
```

## Usage

```python
from hypyxic import compute_hypoxic_burden
hypoxic_burden = compute_hypoxic_burden(sp02_signal, sp02_sr, sleep_stage_annotation, sleep_stage_sr, resp_events_start, resp_events_duration, to_plot=False)
```
## Warning

This implementation output on the test data is 19.361256096995703, the original matlab output is 19.433.
Use at your own risk.

The reason of this difference is that MATLAB code is not 100% convertible to python, since MATLAB is a closed source enviroment. More specifically, funcions like findpeaks are the 100% the same.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.