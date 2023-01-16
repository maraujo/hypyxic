import pandas as pd
import logging
import h5py
from hypyxic import compute_hypoxic_burden


if __name__ == "__main__":
    print("Testing function compute_hypoxic_burden.")
    with h5py.File('data_example\OriginalSignalsExample.mat', 'r') as f:
        resp_events = f["RespEvents"]
        resp_events_start = pd.DataFrame(resp_events["Start"])[0]
        resp_events_duration = pd.DataFrame(resp_events["Duration"])[0]
        
        sleep_stage = f["SleepStage"]
        sleep_stage_annotation = pd.DataFrame(sleep_stage["Annotation"])[0]
        sleep_stage_codes = pd.DataFrame(sleep_stage["Codes"])[0] # {'Wake','Stage 1','Stage 2','Stage 3','Stage 4','REM','Indeterminant'}
        sleep_stage_sr = pd.DataFrame(sleep_stage["SR"])[0][0]
        
        sp02 = f["SpO2"]
        sp02_signal = pd.DataFrame(sp02["Sig"]).T[0]
        sp02_sr = pd.DataFrame(sp02["SR"])[0][0]
        
        hypoxic_burden = compute_hypoxic_burden(sp02_signal, sp02_sr, sleep_stage_annotation, sleep_stage_sr, resp_events_start, resp_events_duration, to_plot=False)
        logging.warning("This implementation output is {}, the original matlab output is {}.\nUse at your own risk.".format(hypoxic_burden, 19.4330))
        
