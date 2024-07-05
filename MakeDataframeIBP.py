# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:45:25 2024
Checking the duration of sensor dysfunction on RR
@author: Anne.vandenElzen
"""

import pandas as pd
import numpy as np
import os

from SensorDysfunction import SensorDysfunction


def expand_list_with_window(indices, half_window):
       expanded_list = set()
       for index in indices:
           for i in range(index - half_window, index + half_window + 1):
               if i>=0:
                   expanded_list.add(i)
       return sorted(expanded_list)
   
   
def get_start_end_points(indices, half_window):
    highlight_window = expand_list_with_window(indices, half_window)
    events = []
    start_idx = 0
    
    diff = np.diff(highlight_window)
    split_points = np.where(diff > 1)[0]
    
    for split_idx in split_points:
        end_idx = split_idx
        events.append([highlight_window[start_idx], highlight_window[end_idx]])
        start_idx = end_idx + 1
        
    events.append([highlight_window[start_idx], highlight_window[-1]])
    
    return events



class MakeDataframeIBP:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def make_dataframe(self, half_window = 60):
        df_events =  pd.DataFrame(columns =['pseudo_id', 'nr_events', 'event_start_time', 'event_end_time','event_start_idx', 'event_end_idx', 'max', 'min','median', 'height', 'nr_nan'])
    
        list_pseudo_id = []
        list_nr_events = []
        list_event_start_time = []
        list_event_end_time = []
        list_event_start_idx = []
        list_event_end_idx = []
        list_scaled_timing = []
        list_total_time = []
        #list_knik_idx = []
        list_max = []
        list_min = []
        list_median = []
        list_height = []
        list_nr_nan = []
    
        grouped = self.dataframe.groupby('pseudo_id')
    
        for pseudo_id, chunk in grouped:
            print(pseudo_id)
            
            chunk = self.dataframe[self.dataframe['pseudo_id']== pseudo_id]
            sdf = SensorDysfunction(chunk)
            blood_sampling = sdf.blood_sampling()
           
            highlight = expand_list_with_window(blood_sampling, half_window)
            chunk.reset_index(inplace = True, drop = True)
            
            if len(highlight)>0: #only check for data with highlights
                events = get_start_end_points(highlight, half_window)
            
                for k in range(len(events)):
                    if events[k][0]<0:
                        start_event = 0
                        segment = chunk.iloc[start_event :events[k][1]]
                    if events[k][1]>chunk.shape[0]:
                        end_event = chunk.shape[0]-1
                        segment = chunk.iloc[events[k][0] :end_event]
                    else:
                        segment = chunk.iloc[events[k][0] :events[k][1]]
                        
                    segment.reset_index(inplace = True, drop = True)
                    
                    #pseudo_id and event count
                    list_pseudo_id.append(pseudo_id)
                    list_nr_events.append(len(events))
                    
                    #stuff about start and end
                    list_event_start_time.append(segment['pat_datetime'].iloc[0])
                    list_event_end_time.append(segment['pat_datetime'].iloc[-1])
                    list_event_start_idx.append(events[k][0]) 
                    list_event_end_idx.append(events[k][1]) 
                    
                    #min and max 
                    #list_knik_idx.append(np.nanargmax(np.diff(np.diff(segment['mon_ibp_mean']))) + 1)
                    list_max.append(np.nanmax(segment['mon_ibp_mean']))
                    list_min.append(np.nanmin(segment['mon_ibp_mean'].min()))
                    list_median.append(np.nanmedian(segment['mon_ibp_mean']))
                    list_height.append(np.nanmax(segment['mon_ibp_mean']) - np.nanmedian(segment['mon_ibp_mean']) )
                    
                    #total number of nan
                    list_nr_nan.append(sum(segment['mon_ibp_sys'].isna()))
            
        df_events["pseudo_id"] = list_pseudo_id
        df_events["nr_events"] = list_nr_events
        df_events["event_start_time"] = list_event_start_time
        df_events["event_end_time"] = list_event_end_time
        df_events["event_start_idx"] = list_event_start_idx
        df_events["event_end_idx"] = list_event_end_idx
        #df_events['scaled_timing'] = list_scaled_timing
        #df_events["knik_idx"] = list_knik_idx
        df_events["max"] = list_max
        df_events["min"] = list_min
        df_events["median"] = list_median
        df_events["height"] = list_height
        df_events["nr_nan"] = list_nr_nan
        
        return df_events


