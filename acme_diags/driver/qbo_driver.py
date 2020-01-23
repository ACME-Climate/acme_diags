from __future__ import print_function

import os
import numpy
import scipy.fftpack
import cdms2
import cdutil
import genutil
import acme_diags
from acme_diags.derivations import acme, default_regions
from acme_diags.driver import utils
from acme_diags.metrics import rmse, corr, min_cdms, max_cdms, mean, std
import matplotlib.pyplot as plt

def process_u_fordiag1(data_region):
    # Average over longtiude
    data_lon_ave = cdutil.averager(data_region, axis = 'x')
    # Average over latitude
    data_lon_lat_ave = cdutil.averager(data_lon_ave, axis = 'y')
    lev_data = data_lon_lat_ave.getAxis(1)
    return data_lon_lat_ave,lev_data


def sg_deseasonNoPercent(xraw):
    #Calculates the deasonsalized data
    # create array to hold climatological values and deseasonalized data
    xclim = numpy.zeros((12,1))
    x_deseasoned = numpy.zeros(xraw.shape)
    for month in numpy.arange(0,12):
        xclim[month] = numpy.nanmean(xraw[month::12])
    for year in numpy.arange(0,int(numpy.floor(len(x_deseasoned)/12))):
        for month in numpy.arange(0,12):
            x_deseasoned[(year)*12+month] = xraw[(year)*12+month]-xclim[month]
    return x_deseasoned

def calc_20to40mo_fftamp(qboN,levelN):
    #Calculates the amplitude of wind variations in the 20 - 40 month period
    pspecN = numpy.zeros(levelN.shape)
    ampN   = numpy.zeros(levelN.shape)
    
    for ilev in numpy.arange(len(levelN)):
        y_input = sg_deseasonNoPercent(numpy.squeeze(qboN[:,ilev]))
        Y = scipy.fftpack.fft(y_input)
        n = len(Y)
        f = numpy.arange(n/2)/n
        period = 1/f
        fyy = Y[0:int(numpy.floor(n/2))]*numpy.conj(Y[0:int(numpy.floor(n/2))])
        # choose the range 20 - 40 months that captures most QBOs (in nature)
        pspecN[ilev] = 2*numpy.nansum(fyy[(period<=40) & (period>=20)])
        ampN[ilev]   = numpy.sqrt(2*pspecN[ilev])*(f[1]-f[0])
    return pspecN,ampN

def nextpow2(x):
    """ Given a number, the function calculates the exponent for the next power of 2
    exp=nextpow2(number)"""
    res = numpy.ceil(numpy.log2(x))
    return res.astype('int')

def process_u_fordiag3(data_region):
    #Average over vertical levels and horizontal area
    levBot=22    #
    levTop=18    #
    data_lat_lon_ave = cdutil.averager(data_region,axis='xy') #average over lat and lon
    x0=numpy.nanmean(numpy.array(data_lat_lon_ave(level=(18,22))),axis=1)#average over vertical
    #x0 should now be 1D
    return x0

def calc_PSD_fromdeseason(xraw,periodNew):
    x0=sg_deseasonNoPercent(xraw)

    Fs0=1 #sampling frequency: assumes frequency of sampling = 1 mo
    T0=1/Fs0
    L0=len(xraw)
    t0=numpy.arange(0,L0)*T0
    NFFT0=2**nextpow2(L0)

    # Apply fft on x0 with n = NFFT
    X0=scipy.fftpack.fft(x0,n=NFFT0)/L0
    f0=Fs0*numpy.arange(0,(NFFT0/2+1))/NFFT0 # frequency (cycles/month)

    period0=1/f0

    AMP0=2*abs(X0[0:int(NFFT0/2+1)])  # amplitude
    PSDx0=AMP0**2/L0             # Power spectral density
    Tend0=T0*L0
    Pxf0=Tend0*numpy.sum(PSDx0)     # Total spectral power
    period0_flipped=period0[::-1]  # need to flip original period, AMP, and PSDx0 because 
    AMP0_flipped=AMP0[::-1]
    PSDx0_flipped=PSDx0[::-1]
    
    AMPnew0 =numpy.interp(periodNew,period0_flipped[:-1],AMP0_flipped[:-1])
    PSDxnew0=numpy.interp(periodNew,period0_flipped[:-1],PSDx0_flipped[:-1])
    return PSDxnew0,AMPnew0

def run_diag(parameter):
    variables = parameter.variables
    #regions = parameter.regions
    region = '5S5N'
    test_data = utils.dataset.Dataset(parameter, test=True)
    ref_data = utils.dataset.Dataset(parameter, ref=True)
    for variable in variables:
        test = test_data.get_timeseries_variable(variable)
        ref = ref_data.get_timeseries_variable(variable)
        qbo_region = default_regions.regions_specs[region]['domain']

        test_region = test(qbo_region)
        ref_region = ref(qbo_region)
        
        # Diag 1: Average over longitude & latitude to produce time,height array of u field:
        qbo1,level1 = process_u_fordiag1(test_region)
        qbo0,level0 = process_u_fordiag1(ref_region)
        
        # Save these transient variables into netcdf format
        
        # Diag 1: Plotting
        label_size=14
        color_levels=numpy.arange(-50,51,100./20.)
        months = qbo0.shape[0]

        fig=plt.figure(figsize=(14,10))
        ax=fig.add_subplot(2,1,2)
        X0,Y0=numpy.meshgrid(numpy.arange(0,months),level0)
        cmap2 = plt.cm.RdBu_r
        ax.invert_yaxis()
        ax.set_title('ERA-interim U 5S-5N',size=label_size,weight='demi')
        ax.set_xlabel('month',size=label_size)
        ax.set_ylabel('hPa',size=label_size)
        plt.yticks(size=label_size)
        plt.xticks(size=label_size)
        plt.yscale('log')
        plt.ylim([100,1])
        pcolor_plot=plt.contourf(X0,Y0,qbo0.T,color_levels,cmap=cmap2)
        cbar0=plt.colorbar(pcolor_plot,ticks=[-50, -25, -5, 5, 25, 50]) 
        cbar0.ax.tick_params(labelsize=label_size)
        
        ax=fig.add_subplot(2,1,1)
        X1,Y1=numpy.meshgrid(numpy.arange(0,months),level1)
        cmap2 = plt.cm.RdBu_r
        ax.invert_yaxis()
        plt.yscale('log')
        plt.yticks(size=label_size)
        plt.xticks(size=label_size)
        plt.ylim([100,1])
        pcolor_plot=plt.contourf(X1,Y1,qbo1.T,color_levels,cmap=cmap2)
        cbar0=plt.colorbar(pcolor_plot,ticks=[-50, -25, -5, 5, 25, 50]) 
        cbar0.ax.tick_params(labelsize=label_size)
        ax.set_title('E3SM U 5S-5N',size=label_size,weight='demi')
        #ax.set_xlabel('month',size=label_size)
        ax.set_ylabel('hPa',size=label_size)
        fig.savefig('qbo_fig1.png')


        # Diag 2: Calculate and plot the amplitude of wind variations with a 20-40 month period
        pspec0,amp0=calc_20to40mo_fftamp(numpy.squeeze(numpy.array(qbo0)),level0)
        pspec1,amp1=calc_20to40mo_fftamp(numpy.squeeze(numpy.array(qbo1)),level1)

        # Save these transient variables into netcdf format


        # Diag 2: Plotting
        

        label_size=14

        fig=plt.figure(figsize=(10,10))
        ax=fig.add_subplot(1,1,1)
        ax.invert_yaxis()
        ax.set_title('QBO Amplitude (period = 20-40 months)',size=label_size,weight='demi')
        ax.set_xlabel('Amplitude (m/s)',size=label_size)
        ax.set_ylabel('Pressure (hPa)',size=label_size)
        plt.yticks(size=label_size)
        plt.xticks(size=label_size)
        plt.yscale('log')
        plt.grid('on')
        plt.ylim([100,1])
        plt.xlim([0,30])
        era_line, =ax.plot(amp0[:],level0[:],'k-o')
        e3sm_line, =ax.plot(amp1[:],level1[:],'r--o')
        ax.legend((era_line,e3sm_line),('ERA-interim','E3SM'),loc='upper right',fontsize=label_size)
        fig.savefig('qbo_fig2.png')


        # Diag 3:Calculate the Power Spectral Density

        #Pre-process data to average over lat,lon,height
        x1=process_u_fordiag3(test_region)
        x0=process_u_fordiag3(ref_region)

        #Calculate the PSD and interpolate to periodNew
        periodNew=numpy.concatenate((numpy.arange(2,33),numpy.arange(34,100,2)),axis=0)#Specify periods to plot
        PSDxnew0,AMPnew0=calc_PSD_fromdeseason(x0,periodNew)
        PSDxnew1,AMPnew1=calc_PSD_fromdeseason(x1,periodNew)

        # Save these transient variables into netcdf format

        #Diag 3: Plotting

        label_size=14

        fig=plt.figure(figsize=(10,7))
        ax=fig.add_subplot(1,1,1)
        ax.set_xlabel('Period (months)',size=label_size,weight='demi')
        ax.set_ylabel('Amplitude (m/s)',size=label_size,weight='demi')
        plt.yticks(size=label_size)
        plt.xticks(size=label_size)
        era_line,=ax.plot(periodNew,AMPnew0,'k-o')
        e3sm_line,=ax.plot(periodNew,AMPnew1,'r--o')
        plt.xlim((0,50))
        plt.grid('on')
        ax.legend((era_line,e3sm_line),('ERA-interim','E3SM'),loc='upper right',fontsize=label_size)
        fig.savefig('qbo_fig3.png')


    
    

    

