from .core_parser import CoreParser
from .zonal_mean_2d_parser import ZonalMean2dParser
from .meridional_mean_2d_parser import MeridionalMean2dParser
from .area_mean_time_series_parser import AreaMeanTimeSeriesParser
from .enso_diags_parser import EnsoDiagsParser
from .qbo_parser import QboParser
from .streamflow_parser import StreamflowParser
from .diurnal_cycle_parser import DiurnalCycleParser

SET_TO_PARSER = {
    'zonal_mean_xy': CoreParser,
    'zonal_mean_2d': ZonalMean2dParser,
    'meridional_mean_2d': MeridionalMean2dParser,
    'lat_lon': CoreParser,
    'polar': CoreParser,
    'cosp_histogram': CoreParser,
    'area_mean_time_series': AreaMeanTimeSeriesParser,
    'enso_diags': EnsoDiagsParser,
    'qbo': QboParser,
    'streamflow': StreamflowParser,
    'diurnal_cycle': DiurnalCycleParser
}

