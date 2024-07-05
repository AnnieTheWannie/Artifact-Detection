# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:09:25 2024

@author: Anne.vandenElzen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

#Check for the path where the SensorDysfunction class is
#os.chdir('Z:/ADS_2024/2 - Sine_Zang_Anne/Anne/codebooks/')
#from SensorDysfunction import SensorDysfunction

#%%





class ChangePointDetector:
    def __init__(self, dataframe):
        self.dataframe = dataframe
      
    
    def extract_data(self, column_name):
        """
        Extract data from the dataframe based on a specific column name.
        
        Returns:
            pd.Series of a given patient and a given parameter (column_name).
        """
        
        if column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
        data = self.dataframe[column_name]
        data.reset_index(inplace = True, drop = True)
        
        return data
    
    def simple_moving_average(self, signal, window_size):
        """
        Simple moving average with as window size window_size.
        
        Returns:
            List with the simple moving average
        """
        sma = []
        for i in range(len(signal)- window_size+1):
            window = signal[i:i+window_size]
            window_average = sum(window)/window_size
            sma.append(window_average)
        return sma
    
    def median_ignore_nan(x):
        return np.nanmedian(x)
     
    def moving_median(self, signal, window_size):
        """
        Simple moving average with as window size window_size.
        
        Returns:
            List with the simple moving average
        """
        half_window = window_size//2
        median_filtered = signal.rolling(window = window_size, center = True).median()
        median_filtered[:half_window] = np.nan
        median_filtered[-half_window:] = np.nan
        
        return median_filtered
        
         
def difference_observed_smoothed(signal, smoothed):
    """
    Compute the difference between and moving averge signal and the original signal
    
    Returns:
        List with signal-avg. If this is not possible for an element it is None.
    """

    difference = []
    for i in range(len(signal)):
        if smoothed[i] is None:
            difference.append(None)
        else:
            diff = signal[i] - smoothed[i]
            difference.append((diff))
    return difference

   
    def diff_geq_threshold(self, diff, threshold):
       
        x = [x for x in diff if abs(x)>threshold]
        return x.index
    
    def change_point_median_experiment(self, signal, tau, eps):
        """
        Own implemented easy approach to find change points
        
        Returns:
            list with all timepoints where the value in the signal deviated more then eps from
            the median of the tau observation before
        """
        change_points = []
        T = signal.shape[0]
        for t in range(tau, T-tau):
            diff = abs(signal.iloc[t] - np.median(signal.iloc[t-tau: t-1]))
            print(diff)
            if diff>eps:
                change_points.append(t)   
        return change_points

    # def ecdf(self, signal, a, b, u):
    #     """
    #     Returns non parametric empirical cummulative disctribution function (ecdf) for a sub-signal that is bounded
    #     by a and b.
        
    #     Returns:
    #         float that indicates the probability that the random process that generated the signal
    #         will take on value less then u
    #     """
            
    #     sub_signal = signal.iloc[a: b]
    #     less_then_u = (sub_signal < u).sum()
    #     equal_to_u = (sub_signal==u).sum()
        
    #     return 1/(b-a)*(less_then_u + 0.5*equal_to_u)
    
    # def log_likelihood(self, signal, a, b, u):
    #     """
    #     Note that for fixed u the emperical cdf will satisfy nF(u)~Bin(n, F(u)). So
    #     the log likelyhood is known that is then the cost
        
    #     Returns:
    #       float
    #     """
    #     p = self.ecdf(signal, a, b, u)*np.log(self.ecdf(signal, a, b, u)) 
    #     q = (1-self.ecdf(signal, a, b, u))*np.log(1-self.ecdf(signal, a, b, u))
    #     return -(b-a)*(p+q)
    
    
    # def cost(self, signal, a, b):
    #     """cost function per segment bounded by a and b
        
    #     Returns:
    #         float that indicates cost of segment bounded by a and b.
    #     """
    #     T = signal.shape[0]
    #     range_u = np.arrange(1, T)
    #     top = self.log_likelihood(signal, a, b, range_u)
    #     bottom = (range_u - 0.5)*(T-range_u + 0.5)
    #     total = top/bottom
    #     return sum(total)
        
        
    # def NMCD(self, signal, a, b):
    #     """calculates segment cost by approximating an integral proposed by Zou et al"""
    #     n = signal.shape[0]
    #     self.log_likelihood(signal, a, b)
        
#%%       
# df = pd.read_csv('Z:/ADS_2024/2 - Sine_Zang_Anne/Anne/datasets/20240423_fys_chd_more_90.csv')


# #%%
# chunk = df[df['pseudo_id']==5]
# cpd = ChangePointDetector(chunk)
# sdf = SensorDysfunction(chunk)
# signal = cpd.extract_data('mon_sat')
# median_5 = cpd.moving_median(signal, 5)
# median_60 = cpd.moving_median(signal, 60)

# plt.figure(figsize = (25, 10))
# plt.plot(median_5, label = 'rolling_median_t = 5')
# plt.plot(median_60, label = 'rolling_median_t = 60')
# plt.legend()
# plt.show()



# #%%
# x = range(len(median_5))

# down = [index + 1 for index, x in enumerate(np.diff(median_5)) if x < 0]
# up = [index + 1 for index, x in enumerate(np.diff(median_5)) if x > 0]
# straight = [index + 1 for index, x in enumerate(np.diff(median_5)) if x == 0]


# #%%

# plt.figure(figsize = (25, 10))
# plt.plot(median)
# plt.scatter(down, median[down], color = 'red')
# plt.scatter(up, median[up], color = 'green')
# plt.scatter(straight, median[straight], color = 'blue')
# plt.xlabel('time in seconds')
# plt.ylabel('sat')
# plt.show()
# #%%
# dips_start = []
# dips_end = []

# #Find dips
# end = 0
# for i in range(1000):
#     if straight[i]+1 in down and straight[i]>end:
#         start = straight[i]
#         #print('start: ' + str(start))
#         boundary = median[start] #voor zelf herstel
       
#         #print("boundary: "+str(boundary))
#         for j in range(i+1, len(straight)):
#             if straight[j] - start > 60: #alleen dips die max 60 sec zijn
#                 break
#             value = median[straight[j]]
            
#             #print('value: '+str(value))
#             if value > boundary:
#                 end = straight[j]
#                 min_val = np.nanmin(median[start+1:end-1])
#                # max_val = np.nanmax(median[start+1:end-1])
#                 if min_val<95 :#check of deze dips wel onder grenswaarde komt
#                     print("end: "+str(end))
#                     dips_start.append(start)
#                     dips_end.append(end)
#                     break
# #%%   
        
    
# #%%
# residu = median_5 - median_60

# plt.figure(figsize = (25, 10))
# plt.plot(median_5[10000:14031])
# plt.plot(median_60[10000:14031])
# plt.plot(residu[10000:14031])
# plt.show()
# #%%
# for i in range(len(dips_start)):
#     segment = median[dips_start[i]: dips_end[i]]
#     segment.reset_index(inplace = True, drop = True)
#     plt.plot(segment)
# plt.show()

# #%%

# plt.figure(figsize = (25, 10))
# plt.plot(median)
# plt.scatter(dips_start, median[dips_start], color = 'red')
# plt.scatter(dips_end, median[dips_end], color = 'green')
# plt.show()
# #%%
# from scipy.interpolate import interp1d

# plt.figure(figsize = (25, 10))
# for i in range(len(dips_start)):
#     segment = median[dips_start[i]: dips_end[i]]
#     rel_time = dips_end[i] - dips_start[i]
#     t = np.linspace(0, 1, len(segment))
#     f = interp1d(t, segment)
#     resampled_seg = f(np.linspace(0,1,100))
#     resampled_seg -= np.nanmax(resampled_seg)
#     resampled_seg /= - np.nanmin(resampled_seg)
#     plt.plot(resampled_seg)
# plt.show()
    
# #%%
# plt.figure(figsize = (25, 10))
# plt.plot(median[dips_start[0]: dips_end[0]])

# plt.show()
# #%%
# plt.figure(figsize = (25, 10))
# plt.plot(median)
# plt.scatter(down, median[down], color = 'red')
# plt.scatter(up, median[up], color = 'green')
# plt.scatter(straight, median[straight], color = 'blue')
# plt.scatter(x, median[x], color = 'orange')
# plt.show()


# error = difference_observed_smoothed(signal, median)
# thres = np.nanpercentile(error, 99)
# highlight = [index for index, value in enumerate(error) if value > thres]
# plt.figure(figsize = (25, 10))
# plt.plot(signal)
# plt.vlines(highlight, ymin = 0, ymax = 100, alpha = 0.1, color = 'red')
# plt.plot(median)
# plt.plot(error)
# plt.show()


#%%

# df_syn = pd.read_csv("Z:/ADS_2024/2 - Sine_Zang_Anne/Dataset/output240524.csv")
# segment_syn = df_syn["SaO2"]
# segment_syn = segment_syn - segment_syn.mean()
# keep_freq = np.fft.fftfreq(segment_syn.shape[0])

# segment = signal.iloc[0:1190]
# normalized = segment - segment.mean()
# segment_fft = np.fft.fft(normalized)
# frequencies = np.fft.fftfreq(normalized.shape[0])

# keep_freq = [0.2, 0.4]
# filtered_fft = np.zeros_like(segment_fft)

# for freq in keep_freq:
#     indices = np.where(np.isclose(np.abs(frequencies), freq))[0]
#     filtered_fft[indices] = segment_fft[indices]
# filtered_signal = np.fft.ifft(filtered_fft)

# plt.figure(figsize = (25, 10))
# plt.plot(filtered_signal)
# plt.show()
# #%%
# plt.figure(figsize = (25,10))
# plt.plot(np.fft.fftshift(np.fft.fftfreq(normalized.shape[0])), np.fft.fftshift(np.abs(np.fft.fft(normalized))))
# plt.show()


#%%

# # #diff_avg = cpd.difference_observed_mean(signal, avg)
# diff = cpd.difference_observed_mean(signal, median)

# plt.figure(figsize = (25, 10))
# # plt.plot(signal, label = 'observerd')
# # #plt.plot(gaussian, label = 'gaussian smoothed')
# plt.plot(median, label = 'median smoothed')
# # #plt.plot(avg, label = 'moving average')
# # #plt.plot(diff_avg, label = 'difference avg')
# plt.plot(diff, label = 'difference gaussian')
# # plt.legend() 
# # plt.title('mon_ibp_sys')
# plt.show()
# # #%%
# print(len(median))
# print(len(np_signal))


# #%%

# plt.figure(figsize = (25, 10))
# plt.plot(signal, label = 'original')
# plt.plot(der1, label = 'first derivative')
# plt.plot(der2, label = 'second derivative')
# plt.scatter(indices, signal.iloc[indices], label = 'cp')

# plt.legend()
# plt.show()










# #%%
# algo = rpt.Pelt(model = 'rank').fit(np_signal)
# result = algo.predict(pen = 20)

# #print(result)
# #%%
# result_1 = result[:-1]
# plt.figure(figsize = (25, 10))
# plt.plot(signal)
# plt.scatter(result_1, signal.iloc[result_1], color = 'red', label = "Change points")
# plt.show()
# #%%
# print(result)

#%%
# #%%
# detector = ChangePointDetector(df)
# signal = detector.extract_data('mon_hr', pseudo_id = 9)

# indices = detector.sensor_dysfunction(signal)
# indices_CP = detector.change_point_median_experiment(signal, tau= 10, eps=5)


# plt.figure(figsize = (25, 10))
# plt.plot(signal, label = 'mon_hr')
# highlighted_indices = signal.iloc[indices]
# highlighted_indices_CP = signal.iloc[indices_CP]
# plt.scatter(highlighted_indices.index, highlighted_indices.values, color = 'red', label = 'sensor dysfunction')
# #plt.scatter(highlighted_indices_CP.index, highlighted_indices_CP.values, color = 'orange', label = 'CP median approach')
# plt.legend()

# plt.show()

# cleaned = detector.remove_sensor_dysfunction(signal)
# plt.figure(figsize = (25, 10))
# plt.plot(cleaned, label = 'mon_hr')
# highlighted_indices = cleaned.iloc[indices]
# plt.scatter(highlighted_indices.index, highlighted_indices.values, color = 'red', label = 'sensor dysfunction')
# plt.legend()

# plt.show()

# #%%

# #plotting emperical cdf
# cdf = []
# T = signal.shape[0]
# for u in range(0, 250):
#     print(detector.ecdf(signal = signal, a= 0, b = T, u = u), u)
#     cdf.append(detector.ecdf(signal = signal, a= 0, b = T, u = u))
    
# plt.plot(np.diff(cdf))
# #%%
# print(cdf)
