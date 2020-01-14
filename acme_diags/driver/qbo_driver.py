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
#start_year = 1980
#end_year = 1990
#start_time = str(start_year)+'-01-15'
#end_time =str(end_year)+'-12-15'
#test_path = '/Users/zhang40/Documents/ACME_simulations/E3SM_v1/U_185001_201312.nc'
#ref_path = '/Users/zhang40/Documents/ACME_simulations/obs_for_e3sm_diags/time-series/ERA-Interim/ua_197901_201612.nc'
#fin = cdms2.open(test_path)
#test = fin('U',time=(start_time,end_time,'ccb'))
#fin.close()
#fin = cdms2.open(ref_path)
#ref = fin('ua',time=(start_time,end_time,'ccb'))

        test_region = test(qbo_region)
        ref_region = ref(qbo_region)
        print(test_region.shape)
        print(ref_region.shape)
        # Average over longitude:
        test_lon_ave = cdutil.averager(test_region, axis = 'x')
        ref_lon_ave = cdutil.averager(ref_region, axis = 'x')
        
        
        # Average over latitude:
        test_lon_lat_ave = cdutil.averager(test_lon_ave, axis = 'y')
        ref_lon_lat_ave = cdutil.averager(ref_lon_ave, axis = 'y')
        qbo1 = test_lon_lat_ave
        qbo0 = ref_lon_lat_ave
        level0=qbo0.getAxis(1)
        level1=qbo1.getAxis(1)
        print(test_lon_ave.shape)
        print(ref_lon_ave.shape)
        
        # Save these transiant variables into netcdf format
        
        # Plotting
        label_size=14
        color_levels=numpy.arange(-50,51,100./20.)
        months = test_lon_ave.shape[0]
        print('total months', months)
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
        #plt.ylim([100,1])
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
        #plt.ylim([100,1])
        pcolor_plot=plt.contourf(X1,Y1,qbo1.T,color_levels,cmap=cmap2)
        cbar0=plt.colorbar(pcolor_plot,ticks=[-50, -25, -5, 5, 25, 50]) 
        cbar0.ax.tick_params(labelsize=label_size)
        ax.set_title('E3SM U 5S-5N',size=label_size,weight='demi')
        #ax.set_xlabel('month',size=label_size)
        ax.set_ylabel('hPa',size=label_size)
        fig.savefig('qbo_fig1.png')





    
    
    
    

    

