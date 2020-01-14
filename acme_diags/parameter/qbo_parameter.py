from .core_parameter import CoreParameter


class QboParameter(CoreParameter):
    def __init__(self):
        super(QboParameter, self).__init__()
        self.ref_timeseries_input = True
        self.test_timeseries_input = True
        self.granulate.remove('seasons')
        

    def check_values(self):
        if not (hasattr(self, 'start_yr') and hasattr(self, 'end_yr')):
            msg = "You need to define both the 'start_yr' and 'end_yr' parameter."
            raise RuntimeError(msg)


