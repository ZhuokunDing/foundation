import numpy as np
import pandas as pd
from djutils import keys, merge, rowproperty, cache_rowproperty, MissingError, U
from foundation.utils import tqdm, logger
from foundation.virtual import utility, stimulus, recording, scan


@keys
class VisualTrials:
    """Visual Trials"""

    @property
    def keys(self):
        return [
            recording.TrialSet,
            recording.TrialFilterSet,
            stimulus.Video,
        ]

    @rowproperty
    def trial_ids(self):
        """
        Returns
        -------
        Tuple[str]
            tuple of keys (foundation.recording.trial.Trial) -- ordered by trial start time
        """
        from foundation.recording.trial import (
            Trial,
            TrialSet,
            TrialVideo,
            TrialBounds,
            TrialFilterSet,
        )

        # all trials
        trials = Trial & (TrialSet & self.item).members

        # filtered trials
        trials = (TrialFilterSet & self.item).filter(trials)

        # video trials
        trials = merge(trials, TrialBounds, TrialVideo) & self.item

        # trial ids, ordered by trial start
        return tuple(trials.fetch("trial_id", order_by="start"))


@keys
class VisualMeasure:
    """Visual Measure"""

    @property
    def keys(self):
        return [
            recording.Trace,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Resample,
            utility.Offset,
            utility.Rate,
            utility.Measure,
            utility.Burnin,
        ]

    @rowproperty
    def measure(self):
        """
        Returns
        -------
        float
            visual response measure
        """
        from foundation.recording.compute.resample import ResampledTrace
        from foundation.stimulus.video import VideoSet
        from foundation.utility.response import Measure
        from foundation.utils.response import Trials, concatenate

        # trial set
        trialset = (recording.TraceTrials & self.item).fetch1()  # all trials

        # videos
        videos = (VideoSet & self.item).members
        videos = videos.fetch("KEY", order_by=videos.primary_key)

        with cache_rowproperty():
            # visual responses
            responses = []

            for video in tqdm(videos, desc="Videos"):
                # trial ids
                trial_ids = (
                    VisualTrials & trialset & video & self.item
                ).trial_ids  # filter trials by TrialFilterSet

                # no trials for video
                if not trial_ids:
                    logger.warning(
                        f"No trials found for video_id `{video['video_id']}`"
                    )
                    continue

                # trial responses
                trials = (ResampledTrace & self.item).trials(trial_ids=trial_ids)
                trials = Trials(trials, index=trial_ids, tolerance=1)

                # append
                responses.append(trials)

        # no trials at all
        if not responses:
            logger.warning(f"No trials found")
            return

        # concatenated responses
        responses = concatenate(*responses, burnin=self.item["burnin"])

        # response measure
        return (Measure & self.item).link.measure(responses)


@keys
class VisualTrialResp:
    """Visual Mean Response Over Repeated Presentation and Corresponding Stimuli"""

    @property
    def keys(self):
        return [
            recording.ScanUnits,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Resize,
            utility.Resolution,
            utility.Resample,
            utility.Offset,
            utility.Rate,
            utility.Burnin,
        ]

    @rowproperty
    def video_response(self):
        """
        Returns
        -------
            videos     : list of np.ndarray [time, height, width, channel] per video
            responses  : list of np.ndarray [trials, time, units] per video
        """
        from foundation.stimulus.video import VideoSet, Video
        from foundation.utils.video import Video as VideoGenerator, Frame
        from foundation.stimulus.resize import ResizedVideo
        from foundation.recording.compute.visual import VisualTrials
        from foundation.utility.resample import Rate

        # trial set
        all_trial_filt = (recording.TrialFilterSet & "not members").proj()
        trialset = dict(
            trialset_id=(
                recording.ScanTrials & (scan.Scan & self.item) & all_trial_filt
            ).fetch1("trialset_id")
        )  # all trials shown in the scan

        # trace set
        traceset = dict(
            traceset_id=(recording.ScanUnits & self.item).fetch1("traceset_id")
        )

        # videos
        video_ids = (VideoSet & self.item).members
        video_ids = video_ids.fetch("KEY", order_by=video_ids.primary_key)

        # unit order
        unit_order = (
            merge(
                recording.TraceSet.Member & traceset,
                recording.ScanUnitOrder & self.item,
            )
        ).fetch("traceset_index", order_by="trace_order")
        with cache_rowproperty():
            # resampled frames of visual stimuli
            frames = []
            # visual responses
            responses = []

            for video_id in tqdm(video_ids, desc="Videos"):
                # trial ids
                trial_ids = (
                    VisualTrials & trialset & video_id & self.item
                ).trial_ids  # filtered trials by video and TrialFilterSet

                # no trials for video
                if not trial_ids:
                    logger.warning(
                        f"No trials found for video_id `{video_id['video_id']}`"
                    )
                    continue

                trial_resps = []
                trial_video_index = []
                for trial_id in trial_ids:
                    # trial responses
                    trial_resps.append(
                        (
                            recording.ResampledTraces
                            & dict(trial_id=trial_id)
                            & traceset
                            & self.item
                        ).fetch1("traces")
                    )  # [time, units], units are ordered by traceset_index
                    # trial stimuli index
                    index = (
                        ResizedVideo * recording.TrialVideo * recording.ResampledTrial
                        & dict(trial_id=trial_id)
                        & self.item
                    ).fetch1("index")
                    trial_video_index.append(index)
                    # videos.append(video[index].astype(np.uint8))

                trial_resps = np.stack(
                    trial_resps, axis=0
                )[
                    ..., unit_order
                ]  # [trials, time, units], units are ordered by unit_id, trials are ordered by trial start
                # responses.append(trial_resps.mean(axis=0))  # [time, units]
                trial_resps = trial_resps[:, self.item["burnin"] :, :]
                responses.append(trial_resps)  # [trials, time, units]
                trial_video_index = np.stack(trial_video_index, axis=0)
                assert np.all(trial_video_index == trial_video_index[0])
                trial_video_index = trial_video_index[0]
                resized_frames = (ResizedVideo & video_id & self.item).fetch1("video")
                frames.append(
                    resized_frames[trial_video_index].astype(np.uint8)[
                        self.item["burnin"] :, ...
                    ]
                )
        assert len(responses) == len(frames)
        return frames, responses
