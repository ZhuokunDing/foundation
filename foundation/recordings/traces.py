import numpy as np
import datajoint as dj
from djutils import link, group, method, row_method, row_property, MissingError
from foundation.utils.logging import logger
from foundation.utils.traces import truncate, fill_nans
from foundation.recordings import trials

pipe_meso = dj.create_virtual_module("pipe_meso", "pipeline_meso")
pipe_eye = dj.create_virtual_module("pipe_eye", "pipeline_eye")
schema = dj.schema("foundation_recordings")


# -------------- Trace --------------

# -- Base --


class TraceBase:
    """Recording Trace"""

    @row_property
    def trace_times(self):
        """
        Returns
        -------
        1D array
            recording trace
        1D array
            recording times of each point in the trace

        IMPORTANT : arrays must be the same length
        """
        raise NotImplementedError()

    @row_property
    def trial_times(self):
        """
        Returns
        -------
        Iterator[tuple[trials.Trial, 1D array]]
            yields
                trial.Trial
                    recording trial
                1D array
                    recording times of each stimulus flips
        """
        raise NotImplementedError()


# -- Types --


@schema
class MesoActivity(TraceBase, dj.Lookup):
    definition = """
    -> pipe_meso.Activity.Trace
    """

    @row_property
    def trace_times(self):
        from foundation.recordings.scan import stimulus_times

        # scan key
        key = ["animal_id", "session", "scan_idx"]
        key = dict(zip(key, self.fetch1(*key)))

        # times on stimulus clock
        times = stimulus_times(**key)

        # activity trace
        trace = (pipe_meso.Activity.Trace & self).fetch1("trace")

        # trim to same length
        trace, times = truncate(trace, times, tolerance=1)

        # imaging delay
        delay = (pipe_meso.ScanSet.UnitInfo & self).fetch1("ms_delay") / 1000

        return trace, times + delay

    @row_property
    def trial_times(self):

        # restrict trials
        key = trials.TrialsLink.ScanTrials * trials.ScanTrials & self
        keys = (trials.Trials & key).trials

        # yield trials
        for key in keys.fetch(dj.key, order_by=keys.primary_key):

            trial = trials.Trial & key
            flips = trial.flips

            yield trial, flips


class ScanBehaviorTraceBase(TraceBase):
    """Scan Behavior Trace --- stimulus time -> behavior time"""

    @row_property
    def trial_times(self):
        from foundation.recordings.scan import stimulus_times, behavior_times
        from foundation.utils.splines import CenteredSpline

        # scan key
        key = ["animal_id", "session", "scan_idx"]
        key = dict(zip(key, self.fetch1(*key)))

        # times on stimulus and behavior clocks
        stim_times = stimulus_times(**key)
        beh_times = behavior_times(**key)

        # stimulus -> behavior time
        times = CenteredSpline(stim_times, beh_times, k=1, ext=3)

        # restrict trials
        key = trials.TrialsLink.ScanTrials * trials.ScanTrials & self
        keys = (trials.Trials & key).trials

        # yield trials
        for key in keys.fetch(dj.key, order_by=keys.primary_key):

            trial = trials.Trial & key
            flips = times(trial.flips)

            yield trial, flips


@schema
class ScanPupilType(dj.Lookup):
    definition = """
    pupil_type          : varchar(64)   # fitted scan pupil type
    pupil_attribute     : varchar(64)   # fitted scan pupil attribute
    """


@schema
class ScanPupil(ScanBehaviorTraceBase, dj.Lookup):
    definition = """
    -> pipe_eye.FittedPupil
    -> ScanPupilType
    """

    @row_property
    def trace_times(self):
        # times of eye trace on behavior clocks
        times = (pipe_eye.Eye & self).fetch1("eye_time")

        # fetch trace based on pupil type and attribute
        pupil_type, pupil_attr = self.fetch1("pupil_type", "pupil_attribute")

        if pupil_type == "circle":
            # fitted pupil circle
            fits = pipe_eye.FittedPupil.Circle & self

            if pupil_attr == "radius":
                # fitted circle radius
                trace = fits.fetch("radius", order_by="frame_id")

            elif pupil_attr in ["center_x", "center_y"]:
                # fitted circle center
                traces = fits.fetch("center", order_by="frame_id")

                if pupil_attr == "center_x":
                    trace = np.array([np.nan if t is None else t[0] for t in traces])
                else:
                    trace = np.array([np.nan if t is None else t[1] for t in traces])

            else:
                # other fitted circle attributes not implemented
                raise NotImplementedError()

        else:
            # other types not implemented
            raise NotImplementedError()

        return trace, times
