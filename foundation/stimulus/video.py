
from djutils import rowproperty, rowmethod
from foundation.virtual.bridge import pipe_stim, pipe_gabor, pipe_dot, pipe_rdk
from foundation.schemas import stimulus as schema


# ---------------------------- Video ----------------------------

# -- Video Base --


class _Video:
    """Video Stimulus"""

    @rowproperty
    def compute(self):
        """
        Returns
        -------
        foundation.stimulus.compute_video.Video (row)
            video stimulus
        """
        raise NotImplementedError()


# -- Video Types --


@schema.lookup
class Clip(_Video):
    definition = """
    -> pipe_stim.Clip
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute_video import Clip

        return Clip & self


@schema.lookup
class Monet2(_Video):
    definition = """
    -> pipe_stim.Monet2
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute_video import Monet2

        return Monet2 & self


@schema.lookup
class Trippy(_Video):
    definition = """
    -> pipe_stim.Trippy
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute_video import Trippy

        return Trippy & self


@schema.lookup
class GaborSequence(_Video):
    definition = """
    -> pipe_stim.GaborSequence
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute_video import GaborSequence

        return GaborSequence & self


@schema.lookup
class DotSequence(_Video):
    definition = """
    -> pipe_stim.DotSequence
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute_video import DotSequence

        return DotSequence & self


@schema.lookup
class RdkSequence(_Video):
    definition = """
    -> pipe_stim.RdkSequence
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute_video import RdkSequence

        return RdkSequence & self


@schema.lookup
class Frame(_Video):
    definition = """
    -> pipe_stim.Frame
    """

    @rowproperty
    def compute(self):
        from foundation.stimulus.compute_video import Frame

        return Frame & self


# -- Video --


@schema.link
class Video:
    links = [Clip, Monet2, Trippy, GaborSequence, DotSequence, RdkSequence, Frame]
    name = "video"
    comment = "video stimulus"


@schema.linkset
class VideoSet:
    link = Video
    name = "videoset"
    comment = "video stimulus set"


# -- Computed Video --


@schema.computed
class VideoInfo:
    definition = """
    -> Video
    ---
    frames          : int unsigned  # video frames
    height          : int unsigned  # video height
    width           : int unsigned  # video width
    channels        : int unsigned  # video channels
    mode            : varchar(16)   # video mode
    period=NULL     : double        # video period (seconds)
    """

    def make(self, key):
        vid = (Video & key).link.compute.video

        key["frames"] = len(vid)
        key["height"] = vid.height
        key["width"] = vid.width
        key["channels"] = vid.channels
        key["mode"] = vid.mode
        key["period"] = vid.period

        self.insert1(key)
