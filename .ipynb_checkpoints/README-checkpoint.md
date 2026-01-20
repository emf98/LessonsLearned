## Welcome!

This repository contains work relevant to E. Fernandez's dissertation research, specifically **Chapters 3 and 4.** 
A more formal and refined version of this folder is planned following the completion of the bulk of this research.

`INFO.txt` is, in essence, a log diary of the work completed in this repository... Yes, it is a mess. 

The main focus of this work is on identifying forecasts of opportunity of stratosphere-troposphere coupling using one-dimensional features from the stratosphere. Primarily, this research explores the applicability of ellipse metrics describing the geometry of the polar vortex.

## Contents

* `ellipse_lstm` contains code relevant to training/running an LSTM model using stratospheric polar vortex ellipse metrics to predict European temperatures.

* `EOF_calculation` contains code to conduct EOF analysis, normalize data/PCA, and train/test RF and LSTM models using the PCA data as predictors for European temperatures.

* `Europe` was the starting point for this repository and contains preliminary code for training and testing the RF model. Organized chaos. 

* `leadtimes` is specifically for **Chapter 4**. This folder trains and tests the RF model, separated by regions and leadtimes (14-, 20-, 30-days).

* `data` contains (some) of the data used in these analyses. It is all ERA-5 data ([Hersbach et al. 2020](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803)).

Code for calculating stratospheric polar vortex ellipse metrics and/or geometries can be found in [this repository](https://github.com/emf98/SPVMD).
