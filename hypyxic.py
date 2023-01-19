import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from scipy.signal import find_peaks

# compute_hypoxic_burden was based on Matlab implementation of Hypoxic Burden made by Philip de Chazal at https://github.com/pdechazal/Hypoxic-Burden. 
def compute_hypoxic_burden(spo2: pd.Series, spo2_sr: int, sleep_stages: pd.Series, sleep_stages_sr: int, respiratory_events_start: pd.Series, respiratory_events_duration: pd.Series, nan_if_empty = True, to_plot=True):        
    # %%%%%%%%%%%
    # %HB calculation - Original Comments
    # %%%%%%%%%%%%%
    # %Steps
    # %   
    # % 1.	Determine the timing of the average event from the event files
    # %       a.	Calculate the average event duration (DurAvg)
    # %       b.	Calculate the average event gap (AvgEventGap)
    # % 2.	Overlay the oxygen saturation signals (SpO2) associated with all respiratory events. 
    # %       The signals are synchronized at the termination of the respiratory events (TimeZero) 
    # %       and the sampling window includes the SpO2 signal from TimeZero -120 seconds to TimeZero +120 seconds. 
    # %       a.	They are averaged to form the ensemble averaged SpO2 curve.
    # %       b.	The ensemble averaged SpO2 curve is filtered with a 0.03Hz low pass filter to form the filtered ensemble averaged SpO2 curve.
    # %       c.	The filtered averaged SpO2 signal is truncated to span the average onset point to the minimum of the average onset of the next event and 90 seconds. 
    # %           It is truncated to TimeZero-DurAvg to TimeZero + the minimum of 90 seconds and AvgEventGap. The resulting signal is referred to as the SpO2 response.
    # % 3.	Determine the start and end point of the search window from the SpO2 response.
    # %       a.	Find minimum point of SpO2 response (Nadir)
    # %       b.	Find maximum difference between start of truncated averaged SpO2 signal and Nadir (MaxDesatOnset). 
    # %       c.	Find last peak at least 75% of amplitude of MaxDesatOnset before the time occurrence of Nadir. This is the start point of the search window (WinStart). 
    # %       d.	Find maximum difference between Nadir and the end of the SpO2 response (MaxDesatOffset). 
    # %       e.	Find first peak at least 75% of amplitude of MaxDesatOnset after the time occurrence of Nadir. This is the end point of the search window (WinFinish). 
    # % 4.	For each event do the following
    # %       a.	Find the pre-event baseline saturation which is defined as the maximum SpO2 during the 100 seconds prior to the end of the event.
    # %       b.	Find the area between pre-event baseline, the SpO2 curve, and WinStart and WinEnd of the search window.
    # %           i.	If any of the SpO2 curve is above the pre-event baseline, then do not add this negative area
    # %           ii.	If event search window overlaps the next event, then do not add the area twice.
    # % 5.	The Hypoxic Burden is defined as the sum event areas divided by the sleep time and has units of %minutes per hour.

    # Arguments
    # spo2: 1hz oxigen saturation signal
    # spo2_sr: should be 1
    # sleep_stages: Should be the same length of spo2 with sleep stage annotation with codings [0 1 2 3 4 5 9] representing {'Wake','Stage 1','Stage 2','Stage 3','Stage 4','REM','Indeterminant'}
    # sleep_stages_sr: should be 1
    # respiratory_events_start: Time in seconds from start of the recording for the start of a respiration event
    # respiratory_events_duration: Duration of the respective event. The length should be the same of respiratory_events_start
    
    hypoxic_burden = np.nan
    if to_plot:
        fig, axs = plt.subplots(3, clear=True)
    
    # Sanity checks
    if spo2_sr != 1 or sleep_stages_sr != 1:
        raise Exception('The SpO2 signal and Sleep stage annotations should be collected at a rate of 1 sample per second.')
    
    if respiratory_events_start.shape[0] != respiratory_events_duration.shape[0]:
        raise Exception('Respiratory events start and duration should have the same size.')
    
    if spo2.shape[0] != sleep_stages.shape[0]:
        raise Exception('SPO2 and Sleep Stages should have the same size.')
    
    if pd.isnull(respiratory_events_start).any() or pd.isnull(respiratory_events_duration).any():
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
        logging.info("There are {} respiratory events.".format(num_events))
        num_spO2_samples = spo2.shape[0]
        
        #1a. Calculate the average event duration
        avg_event_duration = int(np.ceil(respiratory_events_duration.mean()))
        
        #1b. Calculate the average event gap
        avg_event_gap = np.ceil(respiratory_events_start.diff().mean())
        
        #2a. SpO2 signals for each event are averaged to form the ensemble averaged SpO2 curve.
        spo2_in_events = pd.DataFrame(np.nan, index=range(respiratory_events_start.shape[0]), columns=range(240))
        for event_i in range(num_events):
            end_event = round(respiratory_events_start[event_i] + respiratory_events_duration[event_i] + 0.5)
            if (end_event - 120 > 0) and (end_event + 120 < num_spO2_samples):
                spo2_in_event = spo2[end_event - 120 : end_event +120]
                spo2_in_events.loc[event_i, :] = spo2_in_event.values
            else:
                raise Exception("Not implemented: There is no 120s before and after end of event.")
        mean_spo2_in_events = spo2_in_events.mean(skipna=True) # This is the ensemble averaged SpO2 curve
        if spo2_sr == 1: # For now this should be always true.
            # This comment is from the original code.
            # %     %Code to design the FIR lowpass filter with cutoff at 1/30Hz
            # %     N   = 30;        % FIR filter order
            # %     Fp  = 1/30;       % 1/30z passband-edge frequency
            # %     Fs  = 1;       % 1Hz sampling frequency
            # %     Rp  = 0.00057565; % Corresponds to 0.01 dB peak-to-peak ripple
            # %     Rst = 1e-5;       % Corresponds to 100 dB stopband attenuation
            # %
            # %     B = firceqrip(N,Fp/(Fs/2),[Rp Rst],'passedge'); % eqnum = vec of coeffs
            # %     %fvtool(B,'Fs',Fs,'Color','White') % Visualize filter
            B=[0.000109398212241,   0.000514594526374,   0.001350397179936,   0.002341700062534,
                0.002485940327008,   0.000207543145171,  -0.005659450344228,  -0.014258087808069,
                -0.021415481383353,  -0.019969417749860,  -0.002425120103463,   0.034794452821365,
                0.087695691366900,   0.144171828095816,   0.187717212244959,   0.204101948813338,
                0.187717212244959,   0.144171828095816,   0.087695691366900,   0.034794452821365,
                -0.002425120103463,  -0.019969417749860,  -0.021415481383353,  -0.014258087808069,
                -0.005659450344228,   0.000207543145171,   0.002485940327008,   0.002341700062534,
                0.001350397179936,   0.000514594526374,   0.000109398212241]
            
            # Design the filter
            # Helped by: https://dsp.stackexchange.com/questions/11466/differences-between-python-and-matlab-filtfilt-function
            # TODO: For some reason the shape returned by matlab is not the same of the input signal.
            spo2_filtered = filtfilt(B,1, mean_spo2_in_events.values, axis=0,  padtype = 'odd')
            
            #The filtered averaged SpO2 signal is truncated to span the average onset point to the minimum of the average event gap and 90 seconds
            start=int(120*spo2_sr - avg_event_duration*spo2_sr)
            final=int(120*spo2_sr + min(90,avg_event_gap)*spo2_sr) #Limit to 90 seconds

            if to_plot:
                ax = pd.Series(spo2_filtered[start:final]).plot(ax=axs[0])
                pd.Series(mean_spo2_in_events.values[start:final]).plot(ax=ax)
            
            win_start = -1
            win_finish = -1
            
            #3a. Find minimum point of SpO2 response (Nadir)
            nadir_indexes, peaks_props= find_peaks(-spo2_filtered[start:final])
            if len(nadir_indexes) > 0:
                index = (-spo2_filtered[start:final][nadir_indexes]).argmax()
                nadir_index = nadir_indexes[index]
                nadir = spo2_filtered[start:final][nadir_index]
                if nadir_index >= 2 and nadir_index < len(spo2_filtered[start:final]) - 2:
                    if to_plot:
                        ax.scatter(nadir_index, spo2_filtered[start:final][nadir_index], marker="x")
                    
                
                #3b. Find maximum difference between start of truncated averaged SpO2 signal and Nadir (MaxDesatOnset).
                left_peak_indexes, peaks_props = find_peaks(spo2_filtered[start: start+nadir_index])
                left_peaks = spo2_filtered[start: start+nadir_index][left_peak_indexes]
                if len(left_peak_indexes) > 0:
                    max_desat_onset = spo2_filtered[start: start+nadir_index][left_peak_indexes].max()
                    #3c. Find last peak at least 75% of amplitude of MaxDesatOnset before the time occurrence of Nadir. This is the start point of the search window (WinStart).
                    indexes_peaks_max_desat_onset = np.where(left_peaks - nadir > 0.75 * (max_desat_onset - nadir))[0]
                    win_start = left_peak_indexes[indexes_peaks_max_desat_onset[-1]]
                    
                    if to_plot:
                        ax.scatter(win_start, spo2_filtered[start:final][win_start], color="r", marker="<")
                    
                    #3d. Find maximum difference between Nadir and the end of the SpO2 response (MaxDesatOffset).
                    right_peak_indexes, peaks_props = find_peaks(spo2_filtered[start + nadir_index:])
                    right_peaks = spo2_filtered[start+nadir_index:][right_peak_indexes]
                    if len(right_peak_indexes) > 0:
                        max_desat_offset = spo2_filtered[start + nadir_index :][right_peak_indexes].max()
                        #3e.	Find first peak at least 75% of amplitude of MaxDesatOnset after the time occurrence of Nadir. This is the end point of the search window (WinFinish).
                        indexes_peaks_max_desat_offset = np.where(right_peaks - nadir > 0.75 * (max_desat_offset - nadir))[0]
                        win_finish = right_peak_indexes[indexes_peaks_max_desat_offset[0]] + nadir_index
                        if to_plot:
                            ax.scatter(win_finish, spo2_filtered[start:final][win_finish], color="g", marker=">")
                            ax.set_title("Averaged desaturation curve")
                            
                        
        if len(nadir_indexes) == 0 or win_start == -1 or win_finish == -1:
            logging.warning("Using population defaults")
            win_start = avg_event_duration - 5 * int(spo2_sr)
            win_finish = avg_event_duration + 45 * int(spo2_sr)
        
        win_start = win_start - avg_event_duration
        win_finish = win_finish - avg_event_duration
        percent_mins_desat = 0
        limit = 0 #Prevents double counting of areas when events are within window width of each other
        
        segments_area_to_plot = []
        
        for event_i in range(num_events):
            finish = round((respiratory_events_start[event_i] + respiratory_events_duration[event_i])*spo2_sr + 0.5)
            if finish-100*spo2_sr > 0 and finish + win_finish < spo2.shape[0]:
                # %Double count and negative area correction
                # %       4a.	Find the pre-event baseline saturation which is defined as the maximum SpO2 during the 100 seconds prior to the end of the event.
                # %       4b.	Find the area between pre-event baseline, the SpO2 curve, and WinStart and WinEnd of the search window.
                # %           4bi.	If any of the SpO2 curve is above the pre-event baseline, then do not add this negative area
                # %           4bii.	If event search window overlaps the next event, then do not add the area twice.
                
                # As defined by Ali et al: The preevent baseline saturation was defined as the maximum SpO2 during the 100 s prior to the end of the event.
                # Define a range of indices for the last 100 seconds of data
                last_100_seconds = range(finish-int(100*spo2_sr), finish)
                
                # 4a. Find the maximum SpO2 level in the last 100 seconds
                max_spo2_last_100_seconds = spo2[last_100_seconds].max()

                # Define a range of indices for the window of time being analyzed
                analysis_window = range(max(finish + win_start, limit), finish + win_finish)
                
                if to_plot:
                    spo2[analysis_window].reset_index(drop=True).plot(ax=axs[1], linewidth=0.2)
                    segments_area_to_plot.append({
                        "signal" :  spo2[analysis_window],
                        "threshold" : max_spo2_last_100_seconds
                    })
                # Find the area between pre-event baseline, the SpO2 curve, and WinStart and WinEnd of the search window.
                # If any of the SpO2 curve is above the pre-event baseline, then do not add this negative area
                area_below_threshold = (max_spo2_last_100_seconds - spo2[analysis_window]).apply(lambda x: max(x,0)).fillna(0).sum()

                # Calculate the percentage per minute during which the SpO2 level is below the threshold
                percent_aux = (area_below_threshold / (60 * spo2_sr)) 
                percent_mins_desat += percent_aux

                limit = finish + win_finish
        hour_sleep = (~spo2[(sleep_stages > 0) & (sleep_stages < 9)].isnull()).sum() / 3600*sleep_stages_sr
        HB = percent_mins_desat / hour_sleep
        if to_plot:
            axs[1].set_title("All desaturations")  
            index = 0
            for segment in segments_area_to_plot:
                segment["signal"].index = range(index, index+segment["signal"].shape[0])
                ax = segment["signal"].plot(ax=axs[2])
                y0 = [segment["threshold"]] * len(segment["signal"])
                y1 = segment["signal"]
                ax.fill_between(range(segment["signal"].index[0], segment["signal"].index[-1] + 1), y0, y1, where = y0 > y1, alpha = 0.2)
                # ax.axhline(segment["threshold"], xmin=segment["signal"].index[0], xmax=segment["signal"].index[-1], linewidth=0.4)
                ax.plot([segment["signal"].index[0], segment["signal"].index[-1]], [segment["threshold"], segment["threshold"]], linewidth=0.2)
                index = segment["signal"].index[-1]
            
            axs[2].set_title("Individual Hypoxic Burden. Hypoxic Burden={}".format(HB))
            plt.tight_layout()
            plt.show()
        return HB
