from djutils import merge, rowproperty, rowmethod
from foundation.virtual import fnn, recording
from foundation.schemas import tuning as schema


# ------------------------------------ DirResp ------------------------------------

# -- DirResp Interface --


class DirRespType:
    """Tuning Direction"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
            directions : pd.DataFrame, rows are stimulus.compute.video.direction
            responses  : np.ndarray [directions, units]
        """
        raise NotImplementedError()
    

# -- DirResp Types --
    

@schema.lookup
class RecordingDir(DirRespType):
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
        from foundation.tuning.compute.direction import RecordingDir
        return RecordingDir & self
    

# -- DirResp --


@schema.link
class DirResp:
    links = [RecordingDir]
    name = "dir_resp"
    comment = "responses to directional stimuli"


# -- Computed Directional Tuning --
@schema.computed
class DirTuning:
    definition = """
    -> DirResp
    ---
    direction             : longblob        # [directions]
    tuning                : longblob        # [directions, traces]
    """

    @rowmethod
    def compute(self):
        from foundation.tuning.compute.direction import DirTuning
        return DirTuning & self
    

# ------------------------------------ BiVonMisesFit ------------------------------------
@schema.computed
class BiVonMisesFit:
    pass