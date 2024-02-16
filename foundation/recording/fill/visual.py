from djutils import keys, rowproperty, merge, U
from foundation.virtual import recording, stimulus, utility

@keys
class ScanVideoType():
    """Fill videoset for a scan with a specific video type"""
    @property
    def keys(self):
        return [
            recording.ScanRecording,
            recording.TrialFilterSet,
            U('video_type') & stimulus.Video,
        ]

    @rowproperty
    def videoset(self):
        # filter trials
        from foundation.recording.trial import (
            Trial,
            TrialSet,
            TrialVideo,
            TrialFilterSet,
        )
        from foundation.stimulus.video import VideoSet
        # fill videoset_id
        all_trialset_id = dict(
            trialset_id=(recording.ScanRecording & self.item).fetch1("trialset_id")
        )
        # all trials
        trials = Trial & (TrialSet & all_trialset_id).members
        # filtered trials
        trials = (TrialFilterSet & self.item).filter(trials)
        # video_ids of the same video_type that are shown in the scan
        videos = stimulus.Video & (
            merge(trials, TrialVideo, stimulus.Video) & self.item
        ).proj()
        # trial ids, ordered by trial start
        videoset = VideoSet.fill(videos, prompt=False)
        return videoset