#================================================================================================
# This python script calculates monthly NINO34 index, obtained original from  Ji-Woo Lee 
#(jwlee@llnl.gov), April 2016 and later modified and extended to add more ENSO diags by Jill Zhang Nov 2019
#================================================================================================
#
import cdms2 
import cdutil
import cdtime
import genutil


# Compute NINO index

### Set analysis time period
start_year = 1900 ### ARE THEY CORRECT REFERENCE YEAR FOR CLIMATOLOGY?
end_year = 1902   ### Actual end year + 1... 

model_dir = '/Users/zhang40/Documents/ACME_simulations/E3SM_v1/'
filename = 'TS_185001_201312.nc'

start_time = str(start_year)+'-01-15'
end_time =str(end_year)+'-12-15'

# Define Domain area
idxs = ['NINO34', 'NINO3', 'NINO4'] 
domain = {'NINO34':{'llat':-5., 'ulat':5.,  'llon':190., 'ulon':240.},
          'NINO3' :{'llat':-5., 'ulat':5.,  'llon':210., 'ulon':270.},
          'NINO4' :{'llat':-5., 'ulat':5.,  'llon':160., 'ulon':210.},
          }

nidx = 1

fin = cdms2.open(model_dir + filename) 

for idx in idxs[0:nidx]:   
    lat1 = domain[idx]['llat']
    lat2 = domain[idx]['ulat']
    lon1 = domain[idx]['llon']
    lon2 = domain[idx]['ulon']

    # Load variable
    region = cdutil.region.domain(latitude=(lat1,lat2),longitude=(lon1,lon2))
    var = fin('TS', region,time=(start_time,end_time,'ccb'))(squeeze=1)

    # Landmask here? --- NINO index regions are over ocean.. not urgent...

    # Domain average
    region_avg = cdutil.averager(var, axis='xy')

    # Get anomaly from annual cycle climatology
    sst_avg_anomaly = cdutil.ANNUALCYCLE.departures(region_avg)

    # Get linear regression
    #data[idx].slope, data[idx].intercept = genutil.statistics.linearregression(data[idx])
    ### linear regression of data[idx] and np.array(data[idx]) returns different results.. WHY???
    #without running mean
    nino_index = sst_avg_anomaly
    # Running mean
    #runavg = 1 # no running mean
    ##runavg = 3 # 3-month
    #runavg = 5 # 5-month
    ###runavg = 12 # 12-month  ## Even number is now working for now.. size mismathching x2, y2
    ##runavg = 5*12 # 5-year
    #nino_index = genutil.filters.runningaverage(sst_avg_anomaly,runavg)
    print(nino_index.shape)

#t = d.getTime()
#t.units = d.getTime().units
t = var.getTime().asComponentTime()
#t = cdms.createAxis(t,id='time')

fin.close()

filename1 = 'PRECC_185001_201312.nc'
filename2 = 'PRECL_185001_201312.nc'
fin1 = cdms2.open(model_dir + filename1)
fin2 = cdms2.open(model_dir + filename2)

region = cdutil.region.domain(latitude=(-20,20))
prect = fin1('PRECC', region,time=(start_time,end_time,'ccb'))(squeeze=1) +fin2('PRECL', region,time=(start_time,end_time,'ccb'))(squeeze=1)
prect = prect *24.0 *3600.0 * 1000.0 #convert from m/s to mm/day

# Get anomaly from annual cycle climatology
prect_anomaly = cdutil.ANNUALCYCLE.departures(prect)
print(prect_anomaly.shape)

nlat = len(prect_anomaly.getLatitude())
nlon = len(prect_anomaly.getLongitude())

reg_coe = prect_anomaly[0,:,:](squeeze=1)
for ilat in range(nlat):
    print(ilat)
    for ilon in range(nlon):
        slope, intercept = genutil.statistics.linearregression(prect_anomaly[:,ilat,ilon],x = nino_index)
        reg_coe[ilat,ilon] = slope
print(reg_coe.shape)
        
print(len(prect_anomaly.getLatitude()))













