"""
This program calculates time-derivative of geomagnetic field signals B. The effect of applying the proposed signal processing
methods are shown in the output plots. This project helps to calculate d(B)/d(t) without amplifying the noises in the signal.

    Inputs:  geomagnetic field data (B signals)
             The geomagnetic field data in the CSV file are extracted from www.intermagnet.org.
    Outputs: Time derivative of B signals
    
@author: Mohamadreza Ariannik, PhD
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

files=np.array(glob.glob('*.csv'))
### The input data are loaded 
input_B='Input_data_Magnetic_Field_Yellowknife.csv'
### Loading magnetic field data
data=pd.read_csv(input_B)
bx=np.array(data.iloc[:,0])
by=np.array(data.iloc[:,1])
bz=np.array(data.iloc[:,2])
Fs=data.iloc[0,3]

# Deducting mean value of the magnetic field signals, this improves noise reduction capability
mean_deduction=lambda x:x-np.mean(x)
bx,by,bz=mean_deduction(bx), mean_deduction(by),mean_deduction(bz)

### Creating time samples
t=np.array(range(0,len(bx)))
#########################################################
### Denoising magnetic field signals ###################
def denoise_sig(sig):
    from skimage.restoration import denoise_wavelet
    lev=4
    xd=denoise_wavelet(sig,method='VisuShrink',mode='soft',wavelet_levels=lev,wavelet='sym4',rescale_sigma='True')
    #df=pd.DataFrame(xd)
    #df.to_csv (r'C:\Users\maria\Desktop\export_dataframe.csv', index = True, header=False)
    return xd
### Despiking magnetic field signals
def despike_sig(sig):
    import scipy.signal
    if len(sig)>16:
        window_size=7
    else:
        window_size=3
    y2x=scipy.signal.medfilt(sig, kernel_size=window_size)
        
    return y2x
### This function implements the denoising and despiking processes with respect to their priority
def noise_spike(sig,Freq):
    
    if Freq>1/30:        # Median filter is ignored for low frequency signals 
        var_d=np.var(np.diff(sig),ddof=1) # the number of data sets for division is N-1 instead of N
        if var_d<1:
            y1=despike_sig(sig)
            y2=denoise_sig(y1)
        else:
            y1=denoise_sig(sig)
            y2=despike_sig(y1)
    else:
        y2=denoise_sig(sig)        
    return y2

ypx, ypy=noise_spike(bx,Fs), noise_spike(by,Fs)   # This yields the denoised and despiked magnetic field signals

mag_field=np.array([ypx,ypy])
mag_field=np.transpose(mag_field)
f=pd.DataFrame(mag_field)
f.to_csv ('Offline_Modified_B.csv', index=False, header=False)   # Storing denoised and despiked signals
#########################################################
########### Calculating time derivative of B signals ###########
def wavelet_derivative(filtered_y,Fs):
    ### Fs: sampling frequency is Hz
    import pywt
    from continuous_wavelet import cwtm
    ### Extending the signal at both ends to compensate for the transients
    t=np.arange(0,len(filtered_y))
    n_fit=4
    z1=np.polyfit(t[0:n_fit],filtered_y[0:n_fit],1)
    f1=np.poly1d(z1)
    n_added=32
    x_new=np.arange(-n_added,0)
    y_new=f1(x_new)
    filtered_y1=np.hstack((y_new,filtered_y))
    
    z2=np.polyfit(t[len(filtered_y)-n_fit:len(filtered_y)],filtered_y[len(filtered_y)-n_fit:len(filtered_y)],1)
    f2=np.poly1d(z2)

    x_new2=np.arange(len(filtered_y),len(filtered_y)+n_added)
    y_new2=f2(x_new2)
    filtered_y2=np.hstack((filtered_y1,y_new2))
    ### Determinig scale value based on sampling frequency of B signals
    if Fs==1/60:
        wt_scale=2
    elif Fs>=2/60 and Fs<=3/60:
        wt_scale=4
    else:
        wt_scale=8
    wt_name='db1'
    coef,freqs=cwtm(filtered_y2, wt_scale, wt_name)  ### Calculating continuous wavelet transform
    coef=-coef/(wt_scale**(3/2))
    K=1/4
    yw1=coef/(K/Fs)
    yw2=np.delete(yw1,np.arange(0,n_added)) ### Deleting the added (interpolated) data points 
    yw3=np.delete(yw2,np.arange(len(yw2)-n_added,len(yw2)))  
    return yw3

if Fs<1/30:
    dbx=np.diff(ypx)/(1/Fs)
    dby=np.diff(ypy)/(1/Fs)
else: 
    dbx=wavelet_derivative(ypx, Fs)
    dbx=np.real(dbx)
    dby=wavelet_derivative(ypy, Fs)
    dby=np.real(dby)

#########################################################
DB=np.array([dbx,dby])
DB=np.transpose(DB)

f=pd.DataFrame(DB)
f.to_csv ('Offline_derivative.csv', index=False, header=False)  # Storing time derivative of the signals
#########################################################
######## Plotting geomagnetic fields ######################
time_ax=np.arange(len(bx))/60
# Ploting Bx: x component of the time derivative of the geomagnetic field
plt.figure()
plt.plot(time_ax[0:-1],np.diff(bx), 'b',  label = "Raw $d(B_x)$/dt")
plt.plot(time_ax, dbx, 'r', label="Processed $d(B_{x})/dt$")
plt.legend(loc="upper left")
plt.xlabel('Time (min)')
plt.ylabel('B (nT)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.title('Time derivate of geomagnetic signal at Yellowknife station')
plt.show()

# Ploting By: y component of the time derivative of the geomagnetic field
plt.figure()
plt.plot(time_ax[0:-1],np.diff(by), 'b',  label = "Raw $d(B_y)$/dt")
plt.plot(time_ax, dbx, 'r', label="Processed $d(B_{y})/dt$")
plt.legend(loc="upper left")
plt.xlabel('Time (min)')
plt.ylabel('B (nT)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.title('Time derivate of geomagnetic signal at Yellowknife station')
plt.show()
