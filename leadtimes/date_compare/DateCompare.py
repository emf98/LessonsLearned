####################################################
#File generated on 10/6/2025

#There are three definition statements contained here.
#def upload_data(string); return data
    # - this function opens and saves a pickle file as an array
#def three_compare_keys(list1, list2, list3); return result, count1_only, count2_only, count3_only;
    # - this function compares dates in three lists and returns them as a combined and individual count 
#def two_compare_keys(list1, list2); return result, count1_only, count2_only;
    # - as above but for two lists
####################################################

import numpy as np
import math
import xarray as xr 
import pickle 

import collections

def upload_data(string):
    infile = open(string, 'rb') 
    data = pickle.load(infile)
    infile.close()
    return data;

def three_compare_keys(list1, list2, list3):
    #count for list 1
    count1 = collections.Counter() 
    for day in list1:
        count1[day] += 1
    #count for list 2
    count2 = collections.Counter() 
    for day in list2:
        count2[day] += 1
    #count for list 3
    count3 = collections.Counter() 
    for day in list3:
        count3[day] += 1
        
    result = []
    count1_only = []
    #begin comparing dictionaries. 
    for key in count1:
        if key in count2 and key in count3:
            #save the key to look at for composites
            result.append(key)
        if key not in count2 and key not in count3:
            count1_only.append(key)
    
    count2_only =[]
    for key in count2:
        if key not in count1 and key not in count3:
            count2_only.append(key)
    
    count3_only =[]
    for key in count3:
        if key not in count1 and key not in count2:
            count3_only.append(key)
    
    return result, count1_only, count2_only, count3_only;

def two_compare_keys(list1, list2):
    #count for list 1
    count1 = collections.Counter() 
    for day in list1:
        count1[day] += 1
    #count for list 2
    count2 = collections.Counter() 
    for day in list2:
        count2[day] += 1
        
    result = []
    count1_only = []
    #begin comparing dictionaries. 
    for key in count1:
        if key in count2:
            #save the key to look at for composites
            result.append(key)
        if key not in count2:
            count1_only.append(key)
    
    count2_only =[]
    for key in count2:
        if key not in count1:
            count2_only.append(key)
    
    ##I have this set up to look at the SLTM/RF only values too but I may save those for another time. 
    return result, count1_only, count2_only;