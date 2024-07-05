# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:52:37 2024

@author: Anne.vandenElzen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class SensorDysfunction:
    def __init__(self, dataframe):
        self.dataframe = dataframe#Dataframe per patient!
        
    def extract_data(self, column_name):
        """
        Extract data from the dataframe based on a specific column name.
        
        Returns:
            pd.Series of a given patient and a given parameter (column_name).
        """
        data = self.dataframe[column_name]
        data.reset_index(inplace = True, drop = True)
        return data
    
    def blood_sampling(self):
        """
        Make sure you input ibp_mean ibp_sys and ibp_dia!!!!
        
        if ibp_mean > ibp_sys => due to bloodsampling
        Returns:
            list with indices where ibp_mean>ibp_sys
            
        """
        signal = self.extract_data(['mon_ibp_mean', 'mon_ibp_sys'])
        signal['mon_ibp_sys'] = signal['mon_ibp_sys'].ffill()
        x = signal.loc[signal['mon_ibp_mean'] > signal['mon_ibp_sys']]
        return x.index


        
    def sensor_dysfunction(self, column_name, threshold = 25):
        """
        Detects abrupt changes in the signal that are mostly likely due to
        sensor dysfunction.
        Column name is needed because for different parameter there are different conditions
        to be deemed as sensor dysfunction
        
        Returns: 
        - List with indices of possible sensor dysfunction.
       """
        signal = self.extract_data([column_name])
        if column_name == 'mon_hr':
            threshold = 50
            diff = signal.diff(periods = 2, axis = 0)
        else:
            diff = signal.diff(periods = 1, axis = 0)
        #print(diff)
        x = diff.loc[abs(diff[column_name])>threshold]
        return x.index
    
        # if any(column_name in c for c in columns): 
        #     for t in range(1, T):
        #         diff = abs(signal.iloc[t]-signal.iloc[t-1])
        #         if diff > 25:
        #             sensor_dysfuntion.append(t)
        #     return sensor_dysfuntion
        
    # def weird_peak_hr(self):
    #     signal = self.extract_data('mon_hr')
    #     diff = signal.diff(periods = 4, axis = 0)
    #     x = diff.loc[abs(diff['mon_hr'])>25]
    #     return x.index
        
        
    def sensor_dysfunction_rr(self):
        """
        If rr is equal to 0 this is not possible and a sensor dysfunction.
       
        Returns: 
            - List with indices of possible sensor dysfunction for rr.
        """
        signal = self.extract_data(['mon_rr'])
        x = signal.loc[signal['mon_rr']==0]
        return x.index
        
    def add_to_dataframe(self, indices, column_name):
        """Add a dummy for the parameter indicating whether at that time point there was a dysfunction
        
        Returns:
            Updated dataframe
        """
        
        new_column = 'sensor_dys_'+column_name
        self.dataframe[new_column] = False
        self.dataframe[new_column].iloc[indices] = True
        
            
    def plot_sensor_dysfunction(self, column_name, indices):
        signal = self.extract_data(column_name)
        
        plt.figure(figsize = (25, 10))
        plt.plot(signal, label = column_name)
        plt.scatter(indices, signal.iloc[indices], color = 'red', label = 'sensor dysfunction')
        plt.legend()
        plt.show()
        
    def interarrival_time(self, indices):
        """Determine interarrival time, X_i = S_i - S_i-1 and X_0 = 0 and not listed.
        
        Returns:
            List with difference between 
        
        """
        return np.diff(indices)
        
