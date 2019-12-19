#================================================================================================
# This python script calculates monthly NINO34 index, obtained original from  Ji-Woo Lee 
#(jwlee@llnl.gov), April 2016 and later modified and extended to add more ENSO diags by Jill Zhang Nov 2019
#================================================================================================
#
import os
import cdms2 
import cdutil
import cdtime
import genutil
import acme_diags
import numpy
from acme_diags.driver import utils
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy.ma as ma


def run_diag(parameter):
    variables = parameter.variables
    regions = parameter.regions

    # Compute NINO34 index using SST
    # Define Domain area
    domain = {'NINO34':{'llat':-5., 'ulat':5.,  'llon':190., 'ulon':240.},
          'NINO3' :{'llat':-5., 'ulat':5.,  'llon':210., 'ulon':270.},
          'NINO4' :{'llat':-5., 'ulat':5.,  'llon':160., 'ulon':210.},
          }
    idx = 'NINO34'
    lat1 = domain[idx]['llat']
    lat2 = domain[idx]['ulat']
    lon1 = domain[idx]['llon']
    lon2 = domain[idx]['ulon']

    # Load variable
    nino_region = cdutil.region.domain(latitude=(lat1,lat2),longitude=(lon1,lon2))

    test_data = utils.dataset.Dataset(parameter, test=True)
    sst = test_data.get_timeseries_variable('TS')
    sst_NINO = sst(nino_region)
    #sst_NINO = utils.general.select_region(nino_region, sst, land_frac = None, ocean_frac = None, parameter)
   # Domain average
    sst_avg = cdutil.averager(sst_NINO, axis='xy')
    # Get anomaly from annual cycle climatology
    sst_avg_anomaly = cdutil.ANNUALCYCLE.departures(sst_avg)
    nino_index = sst_avg_anomaly
   
    for var in variables:
        print('Variable: {}'.format(var))

        # Get land/ocean fraction for masking.
        # For now, we're only using the climo data that we saved below.
        # So no time-series LANDFRAC or OCNFRAC from the user is used.
        mask_path = os.path.join(acme_diags.INSTALL_PATH, 'acme_ne30_ocean_land_mask.nc')
        with cdms2.open(mask_path) as f:
            land_frac = f('LANDFRAC')
            ocean_frac = f('OCNFRAC')

        for region in regions:
            print("Selected region: {}".format(region))
            test_data = utils.dataset.Dataset(parameter, test=True)
            test = test_data.get_timeseries_variable(var)
            test_domain = utils.general.select_region(region, test, land_frac, ocean_frac, parameter)
            # Average over selected region, and average
            # over months to get the yearly mean.
            cdutil.setTimeBoundsMonthly(test_domain)
            # Get anomaly from annual cycle climatology
            print(test_domain.shape)
            test_anomaly = cdutil.ANNUALCYCLE.departures(test_domain)

            nlat = len(test_anomaly.getLatitude())
            nlon = len(test_anomaly.getLongitude())
            #reg_coe = numpy.empty_like(test_anomaly[0,:,:](squeeze=1))
            reg_coe = test_anomaly[0,:,:](squeeze=1)
            #nlat = 2
            #nlon = 2
            for ilat in range(nlat):
                print(ilat)
                for ilon in range(nlon):
                    slope, intercept = genutil.statistics.linearregression(test_anomaly[:,ilat,ilon],x = nino_index)
                    reg_coe[ilat,ilon] = slope
            print(reg_coe.shape)
            
            #Create netcdf files to save results
            fout = cdms2.open(var +'_nino3_reg_co.nc','w')
            fout.write(reg_coe)
            fout.close()
            #Save enso index

            #Plotting
            variable = reg_coe
            print(variable)

            def add_cyclic(var):
                lon = var.getLongitude()
                return var(longitude=(lon[0], lon[0] + 360.0, 'coe'))
            
            #var = add_cyclic(var)
            lon = variable.getLongitude()
            lat = variable.getLatitude()
            variable = ma.squeeze(variable.asma())
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            plt.contourf(lon, lat, variable,
                         transform=ccrs.PlateCarree())
            
            ax.coastlines()
            
            #plt.show()
            plt.savefig(var+'_nino3_reg_co.png')



            

            
        

             
            





    
## Compute NINO index
#
#### Set analysis time period
#start_year = 1900 ### ARE THEY CORRECT REFERENCE YEAR FOR CLIMATOLOGY?
#end_year = 1902   ### Actual end year + 1... 
#
#model_dir = '/Users/zhang40/Documents/ACME_simulations/E3SM_v1/'
#filename = 'TS_185001_201312.nc'
#
#start_time = str(start_year)+'-01-15'
#end_time =str(end_year)+'-12-15'
#
## Define Domain area
#idxs = ['NINO34', 'NINO3', 'NINO4'] 
#domain = {'NINO34':{'llat':-5., 'ulat':5.,  'llon':190., 'ulon':240.},
#          'NINO3' :{'llat':-5., 'ulat':5.,  'llon':210., 'ulon':270.},
#          'NINO4' :{'llat':-5., 'ulat':5.,  'llon':160., 'ulon':210.},
#          }
#
#nidx = 1
#
#fin = cdms2.open(model_dir + filename) 
#
#for idx in idxs[0:nidx]:   
#    lat1 = domain[idx]['llat']
#    lat2 = domain[idx]['ulat']
#    lon1 = domain[idx]['llon']
#    lon2 = domain[idx]['ulon']
#
#    # Load variable
#    region = cdutil.region.domain(latitude=(lat1,lat2),longitude=(lon1,lon2))
#    var = fin('TS', region,time=(start_time,end_time,'ccb'))(squeeze=1)
#
#    # Landmask here? --- NINO index regions are over ocean.. not urgent...
#
#    # Domain average
#    region_avg = cdutil.averager(var, axis='xy')
#
#    # Get anomaly from annual cycle climatology
#    sst_avg_anomaly = cdutil.ANNUALCYCLE.departures(region_avg)
#
#    # Get linear regression
#    #data[idx].slope, data[idx].intercept = genutil.statistics.linearregression(data[idx])
#    ### linear regression of data[idx] and np.array(data[idx]) returns different results.. WHY???
#    #without running mean
#    nino_index = sst_avg_anomaly
#    # Running mean
#    #runavg = 1 # no running mean
#    ##runavg = 3 # 3-month
#    #runavg = 5 # 5-month
#    ###runavg = 12 # 12-month  ## Even number is now working for now.. size mismathching x2, y2
#    ##runavg = 5*12 # 5-year
#    #nino_index = genutil.filters.runningaverage(sst_avg_anomaly,runavg)
#    print(nino_index.shape)
#
##t = d.getTime()
##t.units = d.getTime().units
#t = var.getTime().asComponentTime()
##t = cdms.createAxis(t,id='time')
#
#fin.close()
#
#filename1 = 'PRECC_185001_201312.nc'
#filename2 = 'PRECL_185001_201312.nc'
#fin1 = cdms2.open(model_dir + filename1)
#fin2 = cdms2.open(model_dir + filename2)
#
#region = cdutil.region.domain(latitude=(-20,20))
#prect = fin1('PRECC', region,time=(start_time,end_time,'ccb'))(squeeze=1) +fin2('PRECL', region,time=(start_time,end_time,'ccb'))(squeeze=1)
#prect = prect *24.0 *3600.0 * 1000.0 #convert from m/s to mm/day
#
## Get anomaly from annual cycle climatology
#prect_anomaly = cdutil.ANNUALCYCLE.departures(prect)
#print(prect_anomaly.shape)
#
#nlat = len(prect_anomaly.getLatitude())
#nlon = len(prect_anomaly.getLongitude())
#
#reg_coe = prect_anomaly[0,:,:](squeeze=1)
##nlat = 1
##nlon = 1
#for ilat in range(nlat):
#    print(ilat)
#    for ilon in range(nlon):
#        slope, intercept = genutil.statistics.linearregression(prect_anomaly[:,ilat,ilon],x = nino_index)
#        reg_coe[ilat,ilon] = slope
#print(reg_coe.shape)
#fin1.close()
#fin2.close()
#        
#print(len(prect_anomaly.getLatitude()))
#fout = cdms2.open('PRECT_nino3_reg_co.nc','w')
#fout.write(reg_coe)
#fout.close()
#
#
##Plotting
#import cartopy.crs as ccrs
#import matplotlib.pyplot as plt
#import os
#import numpy.ma as ma
#
#
#def add_cyclic(var):
#    lon = var.getLongitude()
#    return var(longitude=(lon[0], lon[0] + 360.0, 'coe'))
#
#var = reg_coe
#var = add_cyclic(var)
#lon = var.getLongitude()
#lat = var.getLatitude()
#var = ma.squeeze(var.asma())
#ax = plt.axes(projection=ccrs.PlateCarree())
#
#plt.contourf(lon, lat, var,
#             transform=ccrs.PlateCarree())
#
#ax.coastlines()
#
##plt.show()
#plt.savefig('PRECT_nino3_reg_co.png')










