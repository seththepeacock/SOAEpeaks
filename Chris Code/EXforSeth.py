#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXforSeth.py

Created on Mon Nov 25 16:19:30 2024
@author: pumpkin
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from numpy.fft import rfft
from numpy.fft import irfft

import matplotlib.pyplot as plt
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

# ====================================================
root= '../SOAE Coherence/Data/'         # root path to file
fileN = 'human_TH14RearwaveformSOAE.mat'   # file name
#fileN = 'Other/ve10re01.mat'
# ---
Npts= 256*16;     # tau = length of time window into FFT (ideally be 2^N) [# of points] {256*4}
SR= 44100         # sample rate (see above) [Hz] {default = 44100 Hz}
fPlot= [0.2,6]  # freq. limits for plotting [kHz] {[0.2,6]}
# ---
cPlot= [0,1]  # coherence vert. plot limits {[0,1]}
magL=[-1,1]     # mag limits for plotting {[-5,5]}
markB= 1  # boolean to turn off markers for plots (0=no markers) {1}
downSample = 0  # boolean to downsample the data by 1/2
addNoise= 0    # boolean to add noise to waveform before analysis {0}
windowB=0   # boolean to allow for application of a Hanning window {0}
FTnorm= "forward"   # normalization meth. for FFT
# ====================================================
# ==== bookeeping I (re file loading)
fname = os.path.join(root,fileN)
if (fileN[-3:]=='mat' and fileN[0:5]!='Model'):   # load in data as a .mat file
    data = scipy.io.loadmat(fname)  # loading in a .mat file
    wf= data['wf']   # grab actual data
elif(fileN[-3:]=='mat' and fileN[0:8]=='Model/MC'):  # load Vaclav's model
    data = scipy.io.loadmat(fname)
    wf= data['oae']  # can also run the 'doae'
    SR= 40000  # SR specified by Vaclav
else:   # load in as a .txt file
    wf = np.loadtxt(fname)  # loading in a .xtt file
  # --- markers for plotting (if used)  
if markB==0:
    markA=''
    markB=''
else:
     markA='.'
     markB='+'       
# --- deal w/ cricket file structure (and SR)
# NOTE: files from Natasha contain this header (which is deleted in ./Data/x.txt))
#Source File Name:	F177_LL_FFT_1.5 to 30 khz SOAE 1255_ 23.4Amb_23.74Pla
#Signal:	Time - Vib Velocity - Samples
#Time	Time Signal
#[ s ]	[ m/s ]
#if (wf.shape[1]>1):
if (fileN[0:7]=='cricket'):
    SR= round(1/np.mean(np.diff(wf[:,0])))  # use first column to determine SR
    wf= wf[:,1]  # grab second column  
# ==== downsample by 1/2?
if (downSample==1):
    wf= wf[1::2]  # skip every other point
    SR= SR/2
# ==== add in noise to waveform (useful re model sims?)
if (addNoise==1):
    #wf= wf.flatten()+ float(np.mean(wf.flatten()))*10000*np.random.randn(len(wf))
    wf= wf.flatten()+ float(np.mean(np.abs(wf.flatten())))*0.5*np.random.randn(len(wf))

# --- determine numb. of segments for spectral averaging (and use as much wf as possible)
M= int(np.floor(len(wf)/Npts))  # numb. of time segments
print(f'# of avgs = {str(M-1)} ')
print(f'Delay window length = {1000*Npts/SR} ms')
# --- allocate some buffers
storeM= np.empty([int(Npts/2+1),M]) # store away spectral magnitudes
storeP= np.empty([int(Npts/2+1),M])  # store away spectral phases
storeWF= np.empty([int(Npts),M])  # waveform segments (for time-averaging)
storePDtau= np.empty([int(Npts/2+1),M-2])  # smaller buffer for phase diffs re windows
storePDtheta= np.empty([int(Npts/2),M])  # phase diff re lower freq bin
storeWFcorr= np.empty([int(Npts),M-1])  # phase-corrected wf
storeVS= np.empty([int(Npts/2+1),M-2])  # time-delay coherence (for coherogram)
storeT= np.empty([M-2])  # time array for spectrograms
# ==== bookeeping II
Npts= int(np.floor(Npts))
Nt= len(wf)  # total numb. of time points
t= np.linspace(0,(Nt-1)/SR,Nt)   # time array
df = SR/Npts   # freq bin width
freq= np.arange(0,(Npts+1)/2,1)    # create a freq. array (for FFT bin labeling)
freq= SR*freq/Npts
indxFl= np.where(freq>=fPlot[0]*1000)[0][0]  # find freq index re above (0.2) kHz
indxFh= np.where(freq<=fPlot[1]*1000)[0][-1]  # find freq index re under (7) kHz
# ==== spectral averaging loop
for n in range(0,M):
    indx= n*Npts  # index offset so to move along waveform
    signal= np.squeeze(wf[indx:indx+Npts])  # extract segment
    # =======================
    # option to apply a windowing function
    if (windowB==1):
        signal=signal*np.hanning(len(signal))  
    # --- deal w/ FFT
    spec= rfft(signal,norm=FTnorm)  
    mag= abs(spec)  # magnitude
    phase= np.angle(spec) # phase
    # --- store away vals
    storeM[:,n]= mag  # spectral mags
    storeP[:,n]= phase # raw phases
    storeWF[:,n]= signal  # waveform segment
    storePDtheta[:,n]= np.diff(phase)  # phase diff re adjacent freq bin (i.e., \phi_j^{{\theta}})
    # ==== deal w/ time-delayed phase diff. (re last buffer)
    if (n>=1 and n<=M-2):
        indxL= (n-1)*Npts  # previous segment index 
        tS= t[indxL]   # time stamp for that point
        signalL=  np.squeeze(wf[indxL:indxL+Npts])  # re-extract last segment
        specL= rfft(signalL,norm=FTnorm) 
        phaseL= np.angle(specL)
        # --- grab subsequent time segment (re coherogram)
        indxH= (n+1)*Npts  # previous segment index 
        tSh= t[indxH]   # time stamp for that point
        signalH=  np.squeeze(wf[indxH:indxH+Npts])  # re-extract last segment
        specH= rfft(signalH,norm=FTnorm) 
        phaseH= np.angle(specH)
        # ==== now compute phase diff re last segment (phaseDIFF2) or next (phaseDIFF3)
        phaseDIFF2= phase-phaseL # (i.e., \phi_j^{{\tau}})
        phaseDIFF3= phaseH-phase
        # ==== perform "phase correction" re last time buffer
        corrSP= mag*np.exp(1j*phaseDIFF2)    # 
        corrWF= irfft(corrSP, norm=FTnorm)   # convert to time domain
        # ==== compute vector strength (across freq) for this instance in one of two ways
        # (first method seems to make more sense and yield consistent results)
        if (1==1):
            # 1. compute avg. phase diff over these two intervals and associated VS
            #avgDphi= 0.5*(phaseDIFF2+phaseDIFF3)
            zzA= 0.5*(np.sin(phaseDIFF2)+np.sin(phaseDIFF3))
            zzB= 0.5*(np.cos(phaseDIFF2)+np.cos(phaseDIFF3))
            vsI= np.sqrt(zzA**2 + zzB**2)
        else:
            # 2. Use Roongthumskul 2019 format, but not taking the mag
            # (seems like taking the imaginary part yields most useful)
            Fj= spec/specL
            vsI= np.imag(Fj/abs(Fj))
            #vsI= np.angle(Fj/abs(Fj))/(2*np.pi)
            ##vsI= abs(Fj/abs(Fj))
        # --- store
        storePDtau[:,n-1]= phaseDIFF2 # phase diff re previous time segment (i.e., \phi_j^{{\tau}})
        storeWFcorr[:,n-1]= corrWF # phase-corrected wf
        storeVS[:,n-1]= vsI  # time-delayed coherence (aka Gamma)
        storeT[n-1]= tS  #
        
# ==== tdPC: Phase coherence via vector strength re previous segment
xx= np.average(np.sin(storePDtau),axis=1)
yy= np.average(np.cos(storePDtau),axis=1)
coherence= np.sqrt(xx**2 + yy**2)
# ==== nnPC: Phase coherence via vector strength re adjacent freq bin
xxNN= np.average(np.sin(storePDtheta),axis=1)
yyNN= np.average(np.cos(storePDtheta),axis=1)
coherenceNN= np.sqrt(xxNN**2 + yyNN**2)
freqAVG= freq[1:]- 0.5*np.diff(freq)
# ==== bookeeping III
tP = np.arange(indx/SR,(indx+Npts-0)/SR,1/SR) # time assoc. for segment (only for plotting)
specAVGm= np.average(storeM,axis=1)  # spectral-avgd MAGs
specAVGmDB= 20*np.log10(specAVGm)    # "  " in dB
specAVGp= np.average(storeP,axis=1)  # spectral-avgd PHASEs

# ==== deal w/ processing phase sans vector stength
# --- unwrap phase to compute avgd. \phi_j^{{\theta}}? [has minor effect, but necessary for interpretability]
# unwrapping puts phases differences in [-pi, pi] rather than [-2pi, 2pi]
if (1==1): # {1}
    phaseUWtheta= np.unwrap(storeP, axis=0)  # first unwrap w.r.t. frequency axis
    phaseDtheta= np.diff(phaseUWtheta, axis=0)  #  second compute diff. re adjacent bin
    phaseUWtau= np.unwrap(storeP, axis=1) # ditto w.r.t. time window axis
    phaseDtau = np.diff(phaseUWtau, axis=1)
else:
    phaseDtheta= np.diff(storeP, axis=0)
    phaseDtau = np.diff(storeP, axis=1)
storePDthetaAVG= np.average(np.abs(phaseDtheta),axis=1)  # lastly avg. the abs val. over windows
storePDtauAVG= np.average(np.abs(phaseDtau),axis=1)
# --- time-averaged version (sans phase corr.)
timeAVGwf= np.average(storeWF,axis=1)  # time-averaged waveform
specAVGwf= rfft(timeAVGwf,norm=FTnorm)    # magnitude
specAVGwfDB= 20*np.log10(abs(specAVGwf))  # "  " in dB
# --- time-averaged: phase-corrected version
timeAVGwfCorr= np.average(storeWFcorr,axis=1)  # time-averaged waveform
specAVGwfCorr= rfft(timeAVGwfCorr,norm=FTnorm)  # magnitude
specAVGwfCorrDB= 20*np.log10(abs(specAVGwfCorr))  # "  " in dB
# --- complex-averaged vers. of phase-corrected version
# (alternative reality check re performing the irfft and rfft to get specAVGwfCorr)
specAVGpcS= np.average(storeM[:,1:-1]*np.exp(1j*storePDtau),axis=1)  # 


# === deal w/ loess
sigma= 0.08  # local-ness factor {0.1-0.2}
fit= lowess(specAVGmDB,freq/1000,frac=sigma)


# =============================================
# ==== visualize
plt.close("all")
# ==== ** coherence (tdPC and nnPC; along w/ magnitude)
if 1==1:
    fig5, ax5  = plt.subplots(2,1)
    # --- mags. on top
    ax5[0].plot(freq/1000,specAVGmDB,linestyle='-', marker=markA, 
                   color='k',label='Spectral Avg.')
    ax5[0].plot(freq/1000,fit,'g--',label='loess fit')
    #ax5[0].set_xlabel('Frequency [kHz]')  
    ax5[0].set_ylabel('Magnitude [dB]',fontsize=12)
    ax5[0].set_title(fileN,fontsize=10,loc='right') 
    ax5[0].set_xlim(fPlot)
    ax5[0].grid()
    ax5[0].set_ylim([np.min(specAVGmDB[indxFl:indxFh])+magL[0],
                    np.max(specAVGmDB[indxFl:indxFh])+magL[1]])
    # --- coherence on bottom
    ax5[1].plot(freq/1000,coherence,linestyle='-', 
                   marker=markA,color='k',label=r'$C_{\tau}$')
    ax5[1].plot(freqAVG/1000,coherenceNN,'r',lw=1,linestyle='--',
                   marker=markB,label=r'$C_{\theta}$',markersize=4)
    ax5[1].set_xlabel('Frequency [kHz]',fontsize=12)  
    ax5[1].set_ylabel('Phase Coherence',fontsize=12) 
    ax5[1].grid()
    ax5[1].set_xlim(fPlot)
    ax5[1].set_ylim(cPlot)
    ax5[1].legend(loc="upper right")
    plt.tight_layout()