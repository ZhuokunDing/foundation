import numpy as np
from foundation.virtual import utility, recording, stimulus
from foundation.schemas import recording as schema



@schema.computed
class DotResponses:
    definition = """
    -> recording.ScanUnits
    -> recording.TrialFilterSet
    -> stimulus.VideoSet
    -> utility.Offset
    ---
    dots           : blob@external     # [dots]
    responses      : blob@external     # [dots, traces], traces are ordered by unit_id
    finite         : bool              # all values finite
    """

    def make(self, key):
        from foundation.recording.compute.visual import DotTrialResp
        dots, responses = (DotTrialResp & key).dot_response

        # trace values finite
        finite = np.isfinite(responses).all()

        # insert
        self.insert1(dict(key, dots=dots, responses=responses, finite=bool(finite)))
