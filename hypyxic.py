import pandas as pd
import numpy as np
import logging
import h5py

def compute_hypoxic_burden(spo2: pd.Series, spo2_sr: int, sleep_stages: pd.Series, sleep_stages_sr: int, respiratory_events_start: pd.Series, respiratory_events_duration: pd.Series, nan_if_empty = True):
    hypoxic_burden = np.nan
    
    # Sanity checks
    if spo2_sr != 1 or sleep_stages_sr != 1:
        raise Exception('The SpO2 signal and Sleep stage annotations should be collected at a rate of 1 sample per second.')
    
    if respiratory_events_start.shape[0] != respiratory_events_duration.shape[0]:
        raise Exception('Respiratory events start and duration should have the same size.')
    
    if spo2.shape[0] != sleep_stages.shape[0]:
        raise Exception('SPO2 and Sleep Stages should have the same size.')
    
    if pd.isnull(respiratory_events_start).any() or pd.isnull(resp_events_duration).any():
        raise Exception('Neither start of events or duration of events should be nan.')
    
    if respiratory_events_start.shape[0] == 0:
        logging.warning("There is no respiratory event to compute hypoxic burden.")
        if nan_if_empty == False:
            hypoxic_burden = 0
        return hypoxic_burden
    else:
        if (spo2 > 100).any() or (spo2 < 50).any():
            logging.warning("There are extreme values for SPO2. Check it out. We will ignore them to compute hypoxic burden.")
            spo2[(spo2 < 50) | (spo2 > 100)] = np.nan
        num_events = respiratory_events_start.shape[0]
        num_spO2_samples = spo2.shape[0]
        
        #1a. Calculate the average event duration
        avg_event_duration = resp_events_duration.mean()
        
        #1b. Calculate the average event gap
        avg_event_gap = resp_events_start.diff().mean()
        
        #2a. SpO2 signals for each event are averaged to form the ensemble averaged SpO2 curve.
        spo2_in_events = pd.DataFrame(np.nan, index=range(resp_events_start.shape[0]), columns=range(240))
        for event_i in range(resp_events_start.shape[0]):
            end_event = round(resp_events_start[event_i] + respiratory_events_duration[event_i] + 0.5)
            if (end_event - 120 > 0) and (end_event + 120 < num_spO2_samples):
                spo2_in_event = spo2[end_event - 120 : end_event +120]
                spo2_in_events.loc[event_i, :] = spo2_in_event.values
        mean_spo2_in_events = spo2_in_events.mean(skipna=True, axis=1)
        import ipdb;ipdb.set_trace()
        pass

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
        
        hypoxic_burden = compute_hypoxic_burden(sp02_signal, sp02_sr, sleep_stage_annotation, sleep_stage_sr, resp_events_start, resp_events_duration)
        assert hypoxic_burden == 19.4330
        
