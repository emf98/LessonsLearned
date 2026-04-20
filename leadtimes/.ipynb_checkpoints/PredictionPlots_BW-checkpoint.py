##Code for reducing bulk in plotting files

#file generated 9/19/2025, E. Fernandez

#The following definition statements are contained within this file:

#DATA PROCESSING RELATED:
#extrapolate(data); returns same array with additional column added on
#preprocess_ellipse(input,shift,idx); returns input array
#BWcheckevent_label(posXtest,input2,idx); returns arrays for BW plots
#CScheckevent_label(posXtest,data_array,idx); returns 3D array for cross section plots

#PLOTTING RELATED:
##Box and WHisker Plot
#BWplot(Tpos,Tneg,Fpos,Fneg,metrics_list,loc_str,save_str)
##########################################################################################

#relevant import statements
import numpy as np
import math
import pandas as pd
import xarray as xr 
import pickle 
import matplotlib.pyplot as plt

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from geopy.distance import great_circle
import cartopy.crs as ccrs
from matplotlib.path import Path

#__________daily anomaly calculation__________

def daily_anomaly(target):
    dailymean = np.nanmean(target,axis=1)
    anom = np.zeros_like(target)
    for t in np.arange(target.shape[1]):
         anom[:,t] = target[:,t] - dailymean
    print(anom.shape)
    return anom; 

#_____________data extrapolation_____________

def extrapolate(data):
    data_test = data[..., 0:1] 
    data_test.shape
    data = np.concatenate([data, data_test], axis=-1)
    return data;

#____________process input dataset____________

def preprocess_ellipse(input,shift,idx):
    print("Load data...")
    ##open ellipse metric files
    infile = open("../data/ellipse/wind10_redo.p", 'rb') 
    wind10 = pickle.load(infile)
    infile.close()

    infile = open("../data/ellipse/size10_redo.p", 'rb') 
    size10 = pickle.load(infile)
    infile.close()

    infile = open("../data/ellipse/ratio10_redo.p", 'rb') 
    rat10 = pickle.load(infile)
    infile.close()

    infile = open("../data/ellipse/ephi10_redo.p", 'rb') 
    ephi10 = pickle.load(infile)
    infile.close()

    infile = open("../data/ellipse/cenlat10_redo.p", 'rb')
    cenlat10 = pickle.load(infile)
    infile.close()

    infile = open("../data/ellipse/cenlon10_redo.p", 'rb')
    cenlon10 = pickle.load(infile)
    infile.close()

    infile = open("../data/gph/NA_gph_weightedANOM_100.p", 'rb') 
    gph = pickle.load(infile)
    infile.close()

    infile = open("../data/pv/CAP_pvu_weightedANOM_350.p", 'rb') 
    pv = pickle.load(infile)
    infile.close()
    print("Remove leap year and shift data...")
    #remove leap year
    wind10 = np.delete(wind10[:62],[151],1)
    rat10 = np.delete(rat10[:62],[151],1)
    cenlat10 = np.delete(cenlat10[:62],[151],1)
    cenlon10 = np.delete(cenlon10[:62],[151],1)
    size10 = np.delete(size10[:62],[151],1)
    ephi10 = np.delete(ephi10[:62],[151],1)

    #this is used to change the start date from October 19 to November 2nd... and just reduce the overall time observed. 
    
    wind10 = wind10[:,19+shift:168]
    rat10 = rat10[:,19+shift:168]
    cenlat10 = cenlat10[:,19+shift:168]
    cenlon10 = cenlon10[:,19+shift:168]
    size10 = size10[:,19+shift:168]
    ephi10 = ephi10[:,19+shift:168]
    gph = gph[:62,19+shift:168]
    pv = pv[:62,19+shift:168]
    
    print("Test wind shape:", wind10.shape)
    print(" ")
    print("Removing NaNs ...")
    #remove NaNs
    test_comp = []
    indices = np.isnan(wind10)
    for i in range(0,62):
        for j in range(0,idx):
            if indices[i,j] != False:
                #print(i)
                #print(j)
                #print("True")
                wind10[i,j] = 0
                rat10[i,j] = 0
                cenlat10[i,j] = 0
                cenlon10[i,j] = 0
                size10[i,j] = 0
                ephi10[i,j] = 0
                if i >= 57:
                    test_comp.append((i,j))
                else:
                    continue
    print("Returning final input array.")
    input[:,:,0] = wind10[:,:]
    input[:,:,1] = rat10[:,:]
    input[:,:,2] = cenlat10[:,:]
    input[:,:,3] = cenlon10[:,:]
    input[:,:,4] = size10[:,:]
    input[:,:,5] = ephi10[:,:]
    input[:,:,6] = gph[:,:]
    input[:,:,7] = pv[:,:]
    
    return input;

#____________calculate rates of events____________

#only for box and whisker plots
def BWcheckevent_label(posXtest,input2,idx):
    
    ##reduce input to just the testing data
    nolag_Xtest = input2[52:,:,:]
    nolag_Xtest.shape

    ####now I wanna make these plots SO ... I am adding an index column on to X_test ... full version. 
    ranges = np.array([x for x in range(0,idx*10,1)])
    ranges = ranges.reshape(10,idx) 
    ranges.shape
    
    ##Check whether event is in the desired list (true pos/neg or false pos/neg)
    posXtest_set = set(posXtest)

    pos_corr_events = []
    pos_corr_total_events = []
    
    for i in range(0,10):
        for j in range(0,idx):
            #index for the date being observed
            date_index = ranges[i,j]
            if date_index not in posXtest_set:
                continue
            elif date_index in posXtest_set:
                features = nolag_Xtest[i, j, :]
                pos_corr_events.extend(features)
                pos_corr_total_events.append(0)
                
    ##reshape
    Tpos = np.array(pos_corr_events).reshape(len(pos_corr_total_events),4)
    
    return Tpos;

#___________________________PLOTTING RELATED DEFINITIONS___________________________

def BWplot(Tpos_all, Tneg_all, Fpos_all, Fneg_all, metrics_list, loc_str, save_str):
    import matplotlib.ticker as mticker
    
    n_leads = len(Tpos_all)
    lead_labels = ['14d','20d','30d']  # adjust if needed
    
    ticks = ['True +', 'False -', 'True -', 'False +']
    base_positions = np.array([2, 4, 6, 8])
    
    w = 0.22              # slightly wider boxes
    spread = 1.4          # controls spacing between leads (KEY PARAM)
    
    colors = ["midnightblue","royalblue","mediumvioletred","magenta"]
    
    #progressively darker shading
    alphas = [0.35, 0.6, 0.9]  # light → dark
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    axes = axes.flatten()
    
    for m in range(4):  # 4 metrics
        
        for l in range(n_leads):
            # increased spacing
            offset = (l - (n_leads-1)/2) * w * spread
            
            C_pos = Tpos_all[l][:, m]
            F_neg = Fneg_all[l][:, m]
            C_neg = Tneg_all[l][:, m]
            F_pos = Fpos_all[l][:, m]
            
            data = [C_pos, F_neg, C_neg, F_pos]
            positions = base_positions + offset
            
            bp = axes[m].boxplot(data, positions = positions, widths = w, patch_artist = True)
            
            #color by category, shade by lead
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(alphas[l])
                patch.set_edgecolor('black')
        
        axes[m].set_xticks(base_positions)
        axes[m].set_xticklabels(ticks)
        axes[m].set_ylabel(metrics_list[m], fontsize=16)
        axes[m].tick_params(axis='both', labelsize=14)
    
    # legend for lead shading
    #for l in range(n_leads):
        #axes[0].plot([], [], color='black', alpha=alphas[l], linewidth=6, label=lead_labels[l])
    #axes[0].legend(title="Lead", fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(save_str, bbox_inches='tight')
    plt.show()
    return ;

