# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:09:21 2024

@author: Anne.vandenElzen
"""


import pandas as pd
import numpy as np

from ChangePointDetector import ChangePointDetector



class DipCharacterization:
    def __init__(self, signal, parameter_name, pseudo_id):
        self.signal = signal #should be either SpO2 of a given patient or HR as a pd.Series or other array-like
        self.parameter_name = parameter_name #either mon_sat or mon_hr
        self.pseudo_id = pseudo_id
        self.residu = []
        self.median = []
        self.baseline = []
        self.total_time = len(self.signal)
        self.dip_starts = []
        self.dip_ends = []
        self.begin_below = []
        self.end_below = []
        self.std_res = 0
        self.mean_res = 0
        self.mean_res_d = 0
        self.std_res_d = 0
        self.baseline_mean = 0
        self.baseline_std = 0
        
    def moving_median(self, window_size):
        """
        Centered moving median
 
        """
        half_window = window_size//2
        median_filtered = self.signal.rolling(window = window_size, center = True).median()
        median_filtered[:half_window] = np.nan
        median_filtered[-half_window:] = np.nan
        
        return median_filtered
     
    def set_median_residu(self, small_t = 5, large_t = 300):
        """Use rolling median centered with large t as window to find baseline. 
        Use rolling median with small t to remove some minor noise in the original signal
        
        Returns:
            the median signal with the small t as window
            the median signal with the large t as window
            the difference between these two. The residu.
        """
        self.median = self.moving_median(small_t)
        baseline = self.moving_median(large_t)
        #interpolate baseline
        self.baseline = baseline.interpolate(method = 'linear', limit_area = 'inside')
        self.residu = self.median - self.baseline
        
    

    def get_relative_change_residu(self):
        """Creates different lists with timepoints where the array-like signal is increasing/decreasing or stays the same.
        It furthermore returns a list where the signal starts to increase or starts to decrease. The latter two are usefull to
        detect dips.
        
        Returns:
            three lists with timepoints where is array-like signal is increasing/decreasing or stays the same
            two list with timepoints where the array-like signal starts decreasing or stops increasing.
        
        """
        
        diff = np.diff(self.residu)
        
        down = [index + 1 for index, x in enumerate(diff) if x < 0]
        up = [index + 1 for index, x in enumerate(diff) if x > 0]
        straight = [index + 1 for index, x in enumerate(diff) if x == 0]
        
        start_decrease = np.where((diff[:-1]>=0) & (diff[1:] < 0))[0]+1 #note +1 since np.diff() shifts index by 1
        end_increase = np.where((diff[:-1]> 0) & (diff[1:] <= 0))[0]+1
        
        
        return down, up, straight, start_decrease, end_increase

        
        
    def mean_std_res(self):
        """Get the sample mean and standard deviation of the residu and the sample mean and standard deviation 
        of the numerical derivative"""
        
        mean_res = np.nanmean(self.residu)
        sigma = np.nanstd(self.residu)
        mean_res_d = np.nanmean(np.diff(self.residu))
        sigma_d = np.nanstd(np.diff(self.residu))
        baseline_mean = np.nanmean(self.baseline)
        baseline_std = np.nanstd(self.baseline)
        
        self.mean_res = mean_res
        self.std_res = sigma
        self.mean_res_d = mean_res_d
        self.std_res_d = sigma_d
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std
        
    def dip_edges(self, labda, alpha = 2):
        """
        Find the dips where the minimum value of the residu is below the mean residu - labda*SD
        
        Input:
            threshold should be in forms of how many time the standard deviation of the residu
            
        Returns:
            - dips_start: index of where the dip starts
            - dips_end: index of where the dip ends
            - start_val_lst: value of median signal at the start of the dip
            - min_val_lst: value of the minimum value in the dip
            - end_val_lst: value of teh median sinal at the end of the dip
        
        """
        
        down_res, up_res, straight_res, start_decrease, end_increase = self.get_relative_change_residu()
        self.mean_std_res()
        
        
        thres = self.mean_res - labda*self.std_res
        above_zero_indices = np.where(self.residu >= -alpha)[0]
        below_thresh = self.residu < thres

        begin_below = np.where(np.diff(below_thresh.astype(int)) == 1)[0]+1
        end_below = np.where(np.diff(below_thresh.astype(int)) == -1)[0]


        dist_mat = (begin_below - above_zero_indices[:, None])
        dist_mat[dist_mat < 0] = 9999

        dip_starts = above_zero_indices[np.argmin(dist_mat, axis=0)]

        dist_mat = (begin_below - above_zero_indices[:, None])
        dist_mat[dist_mat > 0] = -9999

        dip_ends = above_zero_indices[np.argmax(dist_mat, axis=0)]
        
        #Edge cases are difficult. So if start and end are the same, remove
        index_same = np.where(dip_starts == dip_ends)[0]
        
        dip_starts = np.delete(dip_starts, index_same)
        dip_ends = np.delete(dip_ends, index_same)
        begin_below = np.delete(begin_below, index_same)
        end_below = np.delete(end_below, index_same)
        
        #for one patient something goes wrong with start values being later then end...
        index_not_possible = np.where(dip_ends - dip_starts <0)
        
        dip_starts = np.delete(dip_starts, index_not_possible)
        dip_ends = np.delete(dip_ends, index_not_possible)
        begin_below = np.delete(begin_below, index_not_possible)
        end_below = np.delete(end_below, index_not_possible)
        
        self.begin_below = begin_below
        self.end_below = end_below
        self.dip_starts = dip_starts
        self.dip_ends = dip_ends
    
    def dip_characteristics_median(self, labda, alpha = 2):
        self.dip_edges(labda, alpha)
        
        df = pd.DataFrame(columns = ['pseudo_id', 'dip_start', 'dip_start_relative', 'dip_end', 'dip_start_val', 'dip_end_val', 'boundary_start','boundary_end', 'min_val', 'argmin', 'begin_below', 'end_below'])
        min_val_lst = []
        argmin_lst = []
        slope_start_min = []
        slope_min_end = []

        df['dip_start'] = self.dip_starts
        df['dip_start_relative'] =self.dip_starts/self.total_time
        df['dip_end'] = self.dip_ends
        df['dip_start_val'] = self.median[self.dip_starts].tolist()
        df['dip_end_val'] = self.median[self.dip_ends].tolist()
        df['boundary_start'] =  (self.baseline[self.dip_starts] - self.mean_res - labda*self.std_res).tolist()
        df['boundary_end'] =  (self.baseline[self.dip_starts] - self.mean_res - labda*self.std_res).tolist()

        for i in range(len(self.dip_starts)):
            segment = self.median[self.dip_starts[i]: self.dip_ends[i]]
            min_val_lst.append(np.nanmin(segment))
            argmin_lst.append(np.nanargmin(segment))
            slope_start_min.append((self.median[self.dip_starts[i]] - np.nanmin(segment))/( - np.nanargmin(segment)))
            slope_min_end.append((np.nanmin(segment) - self.median[self.dip_ends[i]] )/(self.dip_starts[i]+ np.nanargmin(segment) - self.dip_ends[i]))
            

        df['slope_start_min'] = slope_start_min
        df['slope_min_end'] = slope_min_end
        df["min_val"] = min_val_lst
        df['argmin'] = argmin_lst
        df['pseudo_id'] = self.pseudo_id
        
        #incase on dip is detected multiple times. Werid
        df = df.drop_duplicates()
        return df    
    
    def dip_characteristics_residu(self, labda, alpha):
        self.dip_edges(labda)
        
        df = pd.DataFrame(columns = ['pseudo_id', 'dip_start', 'dip_start_relative', 'dip_end', 'dip_start_val', 'dip_end_val', 'boundary_start','boundary_end', 'min_val', 'argmin', 'begin_below', 'end_below'])
        min_val_lst = []
        argmin_lst = []
        slope_start_min = []
        slope_min_end = []

        df['dip_start'] = self.dip_starts
        df['dip_start_relative'] =self.dip_starts/self.total_time
        df['dip_end'] = self.dip_ends
        df['dip_start_val'] = self.residu[self.dip_starts].tolist()
        df['dip_end_val'] = self.residu[self.dip_ends].tolist()
        df['boundary'] =  (- self.mean_res - labda*self.std_res).tolist()
        
        

        for i in range(len(self.dip_starts)):
            segment = self.residu[self.dip_starts[i]: self.dip_ends[i]]
            min_val_lst.append(np.nanmin(segment))
            argmin_lst.append(np.nanargmin(segment))
            slope_start_min.append((self.residu[self.dip_starts[i]] - np.nanmin(segment))/( - np.nanargmin(segment)))
            slope_min_end.append((np.nanmin(segment) - self.residu[self.dip_ends[i]] )/(self.dip_starts[i]+ np.nanargmin(segment) - self.dip_ends[i]))
            

        df['slope_start_min'] = slope_start_min
        df['slope_min_end'] = slope_min_end
        df["min_val"] = min_val_lst
        df['argmin'] = argmin_lst
        df['pseudo_id'] = self.pseudo_id
        
        df = df.drop_duplicates()
        return df    
        
             
    def pt_specific(self):
        df = pd.DataFrame(columns = ['pseudo_id', "mean_res", 'std_res', 'mean_res_d', 'std_res_d', 'total_time', 'nr_dips', 'baseline_mean', 'baseline_std'])
        df['pseudo_id'] = [self.pseudo_id]
        df['mean_res'] = [self.mean_res]
        df['std_res'] = [self.std_res]
        df['mean_res_d'] = [self.mean_res_d]
        df['std_res_d'] = [self.std_res_d]
        df['total_time'] = [self.total_time]
        df['nr_dips'] = [len(self.dip_starts)]
        df['baseline_mean'] = self.baseline_mean
        df['baseline_std'] = self.baseline_std
        
        return df
    


