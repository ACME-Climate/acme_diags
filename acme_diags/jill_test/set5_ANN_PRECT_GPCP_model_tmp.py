import cdms2,cdutil
import vcs
import MV2
import EzTemplate
from metrics.computation.reductions import aminusb
import genutil.statistics
import numpy
import cdutil

def compute_rmse(model, obs):
    rmse = -numpy.infty
    try:
        #weights = cdutil.area_weights(model)
        #rmse = float(genutil.statistics.rms(model, obs, axis='xy', weights=weights))
        rmse = float(genutil.statistics.rms(model, obs, axis='xy', weights='generate'))#area-weighted
    except Exception, err:
        print err
    return rmse

def compute_corr(model, obs):
    corr = -numpy.infty
    try:
        #weights = cdutil.area_weights(model)
        #corr = float(genutil.statistics.correlation(model, obs, axis='xy', weights=weights))
        corr = float(genutil.statistics.correlation(model, obs, axis='xy', weights='generate'))#area-weighted
    except Exception, err:
        print err
    return corr

#reference_data_path='/space1/test_data/obs_for_diagnostics/'  # observation
#test_data_path='/space/golaz1/ACME_simulations/20160520.A_WCYCL1850.ne30_oEC.edison.alpha6_01/pp/clim_rgr/0070-0099/'  # model
reference_data_path='../'
test_data_path='../'

#Pre defined variables
var='PRECT'
season='ANN'
#Below should be read from metadata
mod_name='1850_alpha6_01 (yrs0070-0099)'
obs_name='GPCP (yrs1979-2009)'

#Read in data
reference_data_set='GPCP_ANN_climo.nc'  # observation
test_data_set='20160520.A_WCYCL1850.ne30_oEC.edison.alpha6_01_ANN_climo.nc'  # model

f_obs=cdms2.open(reference_data_path + reference_data_set)
f_mod=cdms2.open(test_data_path + test_data_set)

obs_pr=f_obs('PRECT')
mod_pr=(f_mod('PRECC')+f_mod('PRECL'))*3600.0*24.0*1000.0 #this is derived variable convert m/s to mm/d
mod_pr.units='mm/day'
#print dir(mod_pr)

#For plotting, original grid is plotted for model observation, differece plot is regridded to coaser grid. Need if statement to evaluate grid size. aminusb_2ax from uvcmetrics takes care of this,which also considers complex corner cases.
axes1=mod_pr.getAxisList() 
axes2=obs_pr.getAxisList() 
if len(axes1[1])<=len(axes2[1]): #use nlat to decide data resolution, higher number means higher data resolution. For the difference plot, regrid toward lower resolution
    model_grid=mod_pr.getGrid()
    mod_pr_reg=mod_pr
    obs_pr_reg=obs_pr.regrid(model_grid,regridTool='esmf',regridMethod='linear')
else:
    obs_grid=obs_pr.getGrid()
    obs_pr_reg=obs_pr
    mod_pr_reg=mod_pr.regrid(obs_grid,regridTool='esmf',regridMethod='linear')
dif_pr=mod_pr_reg-obs_pr_reg

#calculate metrics and pass in as mv attribute,failed
#obs_pr.mean=round(cdutil.averager(obs_pr, axis='xy', weights='generate'),2) #area-weighting
#mod_pr.mean=round(cdutil.averager(mod_pr, axis='xy', weights='generate'),2) #area-weighting
#dif_pr.mean=round(cdutil.averager(dif_pr, axis='xy', weights='generate'),2) #area-weighting
#
#obs_pr.max=round(mod_pr.max(),2)
#mod_pr.max=round(mod_pr.max(),2)
#dif_pr.max=round(dif_pr.max(),2)
#
#obs_pr.min=round(mod_pr.min(),2)
#mod_pr.min=round(mod_pr.min(),2)
#dif_pr.min=round(dif_pr.min(),2)


#CORR and RMSE need to be calculated after reduction to ensure same array shapes.
rmse= 'RMSE:'+'%.2f' %compute_rmse(obs_pr_reg, mod_pr_reg)
corr= 'CORR:'+'%.2f' %compute_corr(obs_pr_reg, mod_pr_reg)
print rmse, corr

#Plotting
x = vcs.init(bg=True, geometry=(1212,1628))
x.portrait()

x.scriptrun('plot_set_5.json')
template_0 = x.gettemplate('plotset5_0_x_0')
template_1 = x.gettemplate('plotset5_0_x_1')
template_2 = x.gettemplate('plotset5_0_x_2')

#It turns out the long_name attribute of the mv appears as title in .json.
mod_pr.long_name='model'
obs_pr.long_name='observation'
dif_pr.long_name='model-observation'

#Or we can change title attribute, adding 3 more lines though...
#mod_pr.title='model'
#obs_pr.title='observation'
#dif_pr.title='model-observation'
#
#del mod_pr.long_name
#del obs_pr.long_name
#del dif_pr.long_name
template_0.title.priority=1
template_1.title.priority=1
template_2.title.priority=1

template_0.units.priority=1
template_1.units.priority=1
template_2.units.priority=1

#The dataname part should pass in basic model info: model_version...and basic obs info:obsname_version_years
#Change the dataname via var.id. Ideally variable attribute should match vcs attribute.

mod_pr.id=mod_name
obs_pr.id=obs_name
template_0.dataname.priority=1
template_1.dataname.priority=1

isofill = x.createisofill()
isofill.levels=[0,0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17]
#x.setcolormap('rainbow')
x.plot(mod_pr, template_0, isofill)

isofill = x.createisofill()
isofill.levels=[0,0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17]

#x.plot(obs_pr, template_1, isofill)
x.plot(obs_pr, template_1, isofill)
#x.setcolormap('bl_to_darkred')

isofill = x.createisofill()
isofill.levels=[-10,-8, -6, -4, -3, -2, -1,-0.5, 0, 0.5, 1, 2, 3, 4, 6, 8,10]
x.plot(dif_pr, template_2, isofill, comment1=rmse, comment2=corr)

#Create main title for the 3 plots 
main_title=vcs.createtext()
main_title.string=' '.join([var,season])
main_title.x=0.42
main_title.y=0.98
x.plot(main_title)

x.png('test.png')


