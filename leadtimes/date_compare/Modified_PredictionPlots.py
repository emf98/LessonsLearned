##Code for reducing bulk in plotting files

#file generated 9/19/2025, E. Fernandez

#The following definition statements are contained within this file:

#DATA PROCESSING RELATED:
#extrapolate(data); returns same array with additional column added on
#preprocess_ellipse(input,shift,idx); returns input array
#BWcheckevent_label(posXtest,input2,idx); returns arrays for BW plots
#CScheckevent_label(posXtest,data_array,idx); returns 3D array for cross section plots

#PLOTTING RELATED:
#combine_cross(GPH_cpos,GPH_cneg,vert_GPH_cpos,vert_GPH_cneg,TEMP_cpos,TEMP_cneg,
                  #GPHA_cpos,GPHA_cneg, colorbarMin1, colorbarMax1, colorspace1,
                  #colorbarMin2, colorbarMax2, colorspace2, colorbarMin3, colorbarMax3,
                  #colorspace3,lead, save_loc)

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
    infile = open("../../data/ellipse/wind10_redo.p", 'rb') 
    wind10 = pickle.load(infile)
    infile.close()

    infile = open("../../data/ellipse/size10_redo.p", 'rb') 
    size10 = pickle.load(infile)
    infile.close()

    infile = open("../../data/ellipse/ratio10_redo.p", 'rb') 
    rat10 = pickle.load(infile)
    infile.close()

    infile = open("../../data/ellipse/ephi10_redo.p", 'rb') 
    ephi10 = pickle.load(infile)
    infile.close()

    infile = open("../../data/ellipse/cenlat10_redo.p", 'rb')
    cenlat10 = pickle.load(infile)
    infile.close()

    infile = open("../../data/ellipse/cenlon10_redo.p", 'rb')
    cenlon10 = pickle.load(infile)
    infile.close()

    infile = open("../../data/gph/NA_gph_weightedANOM_100.p", 'rb') 
    gph = pickle.load(infile)
    infile.close()

    infile = open("../../data/pv/CAP_pvu_weightedANOM_350.p", 'rb') 
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

##for cross sections
def CScheckevent_label(posXtest,data_array,idx):
    
    ##get data array lat/lon dimensions
    array_shape = data_array.shape
    lat_dim = array_shape[2]
    lon_dim = array_shape[3]

    ranges = np.array([x for x in range(0,idx*10,1)])
    ranges = ranges.reshape(10,idx) 
    ranges.shape
    
    ##Check whether event is in the desired list (true pos/neg or false pos/neg)
    posXtest_set = set(posXtest)

    pos_corr_total_events = []
    data_list = []
    
    for i in range(0,10):
        for j in range(0,idx):
            #index for the date being observed
            date_index = ranges[i,j]
            if date_index not in posXtest_set:
                continue
            elif date_index in posXtest_set:
                pos_corr_total_events.append(0)
                data_list.extend(data_array[i,j,:,:])
                
    ##reshape
    data_list = np.array(data_list).reshape(len(pos_corr_total_events),lat_dim,lon_dim)
    
    return data_list;

#___________________________PLOTTING RELATED DEFINITION___________________________

########################################################################
#Combination of Pos/Neg
def combine_cross(GPH_cpos,GPH_cneg,vert_GPH_cpos,vert_GPH_cneg,TEMP_cpos,TEMP_cneg,
                  GPHA_cpos,GPHA_cneg, colorbarMin1, colorbarMax1, colorspace1,
                  colorbarMin2, colorbarMax2, colorspace2, colorbarMin3, colorbarMax3,
                  colorspace3,lead, save_loc):
    
    lat = np.arange(90, 38, -2)
    lon = np.arange(0, 362, 2)
    lev = np.array([1., 2., 3., 5., 7., 10., 20., 30., 50., 70., 100., 125., 150., 175., 200., 225., 250., 300., 350., 400., 
                450., 500., 550., 600., 650., 700., 750., 775., 800., 825., 850., 875., 900., 925., 950., 975., 1000.])

    fs = 21
    fig = plt.figure(figsize=(20, 19))
    plt.suptitle("Composites of Features during Shared Confident and Correct Predictions from Europe and Canada, "+str(lead),fontsize=26)   
    titles = ["Correct Positive", "Correct Negative"]
    data = [GPH_cpos,GPH_cneg,vert_GPH_cpos,vert_GPH_cneg,TEMP_cpos,TEMP_cneg]
    data1 = [GPHA_cpos,GPHA_cneg]

    # Create axes individually to allow mixing projections
    axes = []
    for i in range(6):
        if i in [0, 1, 4, 5]:
            ax = fig.add_subplot(3, 2, i+1, projection=ccrs.NorthPolarStereo())
        else:
            ax = fig.add_subplot(3, 2, i+1)
        axes.append(ax)

    for i in range(6):
        ax = axes[i]

        if i == 0 or i == 1:
            ax.set_title("10hPa GPH " + str(titles[i]), fontsize=fs-1, y=1.01, x=0.5)
            ax.coastlines()

            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)

            clevel = np.arange(colorbarMin1, colorbarMax1 + colorspace1, colorspace1)

            h = ax.contourf(lon, lat, data1[i], clevel, transform=ccrs.PlateCarree(), cmap="RdBu_r")

            highlight_level1 = 30000
            level_cont = np.arange(28500, 32500, 250)

            for level in level_cont:
                linestyle = 'dashed'
                linewidth = 1.0
                color = 'black'
                if level == highlight_level1:
                    linestyle = 'solid'
                    linewidth = 2.5

                g_cont = ax.contour(
                    lon, lat, data[i], levels=[level],
                    transform=ccrs.PlateCarree(),
                    colors=[color], linestyles=[linestyle], linewidths=[linewidth]
                )
                labels = ax.clabel(g_cont, inline=True, fontsize=12)
                for text in labels:
                    text.set_backgroundcolor("white")
                    text.set_bbox(dict(facecolor="white", edgecolor="none", pad=.35))

            cbar = fig.colorbar(h, ax=ax, orientation="vertical", shrink=0.75, pad=0.05, aspect=30)
            cbar.ax.tick_params(labelsize=fs - 2)

        elif i == 2 or i == 3:
            clevel = np.arange(colorbarMin2, colorbarMax2 + colorspace2, colorspace2)
            ax.set_title("40-80$^o$N GPH, " + str(titles[i - 2]), fontsize=fs-1, y=0.99) 

            h = ax.contourf(
                lon[:180],
                lev[5:],
                data[i][5:],
                clevel,
                cmap="RdBu_r",
                extend="both",
            )
            cbar = plt.colorbar(h, ax=ax, orientation="vertical", shrink=1, fraction=0.1, pad=0.1, aspect=40)
            cbar.ax.tick_params(labelsize=fs - 2)

            ax.tick_params(labelsize=fs - 2)
            ax.set_yscale('log')
            ax.invert_yaxis()
            ax.set_ylabel('Pressure (hPa)', fontsize=fs - 3)
            ax.set_yticks([10, 30, 100, 200, 300, 450, 700, 1000])
            ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xlim(0, 360)
            ax.set_xlabel('Longitude', fontsize=fs - 3)

        elif i == 4 or i == 5:
            ax.set_title("Temp Anoms, " + str(titles[i - 4]), fontsize=fs-1, y=1.01, x=0.5)
            ax.coastlines()

            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)

            level = np.arange(colorbarMin3, colorbarMax3 + colorspace3, colorspace3)

            h = ax.contourf(lon, lat, data[i], level, transform=ccrs.PlateCarree(), cmap="RdBu_r")

            cbar = fig.colorbar(h, ax=ax, orientation="vertical", shrink=0.75, pad=0.05, aspect=30)
            cbar.ax.tick_params(labelsize=fs - 2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(str(save_loc), bbox_inches='tight')
    plt.show()

    
    return ;