from djutils import merge, rowproperty, rowmethod
from foundation.virtual import fnn, recording
from foundation.schemas import tuning as schema


# ------------------------------------ Dot ------------------------------------
@schema.lookup
class DotOnOff:
    definition = """
    on           : bool     # True for on, False for off
    """

# ---------------------------- DotResponse ----------------------------

# -- DotResponse Interface --


class DotResponseType:
    """Tuning Dot"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
            dots       : pd.DataFrame, rows are stimulus.compute.video.dot
            responses  : np.ndarray [dots, units]
        """
        raise NotImplementedError()
    

# -- DotResponse Types --
    

@schema.lookup
class RecordingDot(DotResponseType):
    definition = """
    -> recording.ScanUnits
    -> recording.TrialFilterSet
    -> stimulus.VideoSet
    -> utility.Resize
    -> utility.Resolution
    -> utility.Resample
    -> utility.Offset
    -> utility.Rate
    -> utility.Burnin
    -> utility.Offset.proj(tuning_offset_id="offset_id")
    """

    @rowproperty
    def compute(self):
        from foundation.tuning.compute.dot import RecordingDot
        return RecordingDot & self
    

# -- DotResponse --
    

@schema.link
class DotResponse:
    links = [RecordingDot]
    name = "dot_response"
    comment = "dot response"


# -- Computed Dot STA --
@schema.computed
class DotSta:
    definition = """
    -> DotResponse
    -> DotOnOff
    ---
    sta             : longblob     # [height, width, traces], traces are ordered by unit_id
    pred_corr       : longblob     # [traces], traces are ordered by unit_id
    dot_coverage    : longblob     # [height, width], binary mask of the coverage of the dot
    """

    def make(self, key):
        from foundation.tuning.compute.dot import DotSta
        key['sta'], key['pred_corr'], key['dot_coverage']  = (DotSta & key).sta
        self.insert1(key)


# ---------------------------- Gaussian2DFit ----------------------------
@schema.computed
class DotStaGaussian2DFit:
    definition = """
    -> DotSta
    ---
    mu_x            : longblob     # x-coordinate of the center of the gaussian [-1, 1]
    mu_y            : longblob     # y-coordinate of the center of the gaussian [-1, 1]
    sigma_x         : longblob     # standard deviation of the gaussian in the x-direction
    sigma_y         : longblob     # standard deviation of the gaussian in the y-direction
    theta           : longblob     # rotation angle of the gaussian (0 - np.pi)
    amplitude       : longblob     # amplitude of the gaussian
    offset          : longblob     # offset of the gaussian
    fit_pred_corr   : longblob     # fit prediction correlation
    """

    def make(self, key):
        from foundation.tuning.compute.dot import DotSta
        fit_result = (DotSta & key).sta_gaussian2d_fit
        for k, v in fit_result.items():
            key[k] = v
        self.insert1(key)

