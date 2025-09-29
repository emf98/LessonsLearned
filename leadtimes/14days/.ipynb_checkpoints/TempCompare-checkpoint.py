##Code for conducting temperature anomaly date comparisons and plotting them

#file generated 9/29/2025, E. Fernandez

#The following definition statements are contained within this file:
    #compare_keys(list1, list2); returns count comparison between the two lists
    #Tempcheckevent_label(posXtest,temp,data_array,idx,truth_str); returns arrays for true and neutral temps

#relevant import statements
import numpy as np
import math
import pandas as pd
import xarray as xr 
import pickle 
import collections
#_________________________________________________________________
#Compare dates from lists.
##this instance is just to save the keys so that I can look at composites ... for those dates. 
def compare_keys(list1, list2):
    list1_count = collections.Counter() ##list1 counter dictionary
    for day in list1:
        list1_count[day] += 1
    list2_count = collections.Counter() ##list2 counter dictionary
    for day in list2:
        list2_count[day] += 1
        
    result = []
    list1only = []
    #begin comparing RF and LSTM dictionaries. 
    for key in list1_count:
        if key in list2_count:
            #save the key to look at for composites
            result.append(key)
        if key not in list2_count:
            list1only.append(key)
    
    list2only =[]
    for key in list2_count:
        if key not in list1_count:
            list2only.append(key)
    
    ##I have this set up to look at the SLTM/RF only values too but I may save those for another time. 
    return result, list1only, list2only;

#_________________________________________________________________
#Check and create event labels as either a true temp anomaly of false
def Tempcheckevent_label(posXtest,temp,data_array,idx,truth_str):
    
    ##get data array lat/lon dimensions
    array_shape = data_array.shape
    lat_dim = array_shape[2]
    lon_dim = array_shape[3]

    ranges = np.array([x for x in range(0,idx*10,1)])
    ranges = ranges.reshape(10,idx) 
    ranges.shape
    
    ##Check whether event is in the desired list (true pos/neg or false pos/neg)
    posXtest_set = set(posXtest)

    poscorr_indices = []
    negcorr_indices = []
    neutcorrP_indices = []
    data_list_true = []
    data_list_neut = []
    
    if truth_str == "True Positive" or truth_str == "False Negative": 
        for i in range(0,10):
            for j in range(0,idx):
                #index for the date being observed
                date_index = ranges[i,j]
                if date_index not in posXtest_set:
                    continue
                value = temp[i, j]
                if value == 0:
                    negcorr_indices.append(date_index)
                elif value == 1:
                    poscorr_indices.append(date_index)
                    data_list_true.extend(data_array[i,j,:,:])
                elif value == 2:
                    neutcorrP_indices.append(date_index)
                    data_list_neut.extend(data_array[i,j,:,:])
                    
        count, true, neut = compare_keys(poscorr_indices, neutcorrP_indices)
        ##reshape
        data_list_true = np.array(data_list_true).reshape(len(poscorr_indices),lat_dim,lon_dim)
        data_list_neut = np.array(data_list_neut).reshape(len(neutcorrP_indices),lat_dim,lon_dim)
                    
    elif truth_str == "True Negative" or truth_str == "False Positive": 
        for i in range(0,10):
            for j in range(0,idx):
                #index for the date being observed
                date_index = ranges[i,j]
                if date_index not in posXtest_set:
                    continue
                value = temp[i, j]
                if value == 0:
                    negcorr_indices.append(date_index)
                    data_list_true.extend(data_array[i,j,:,:])
                elif value == 1:
                    poscorr_indices.append(date_index)
                elif value == 2:
                    neutcorrP_indices.append(date_index)
                    data_list_neut.extend(data_array[i,j,:,:])
        
        count, true, neut = compare_keys(negcorr_indices, neutcorrP_indices)
        ##reshape
        data_list_true = np.array(data_list_true).reshape(len(negcorr_indices),lat_dim,lon_dim)
        data_list_neut = np.array(data_list_neut).reshape(len(neutcorrP_indices),lat_dim,lon_dim)
    
    return data_list_true, data_list_neut;
