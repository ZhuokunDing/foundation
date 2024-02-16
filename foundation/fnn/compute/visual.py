import numpy as np
from itertools import repeat
from djutils import keys, rowproperty, cache_rowproperty
from foundation.utils import tqdm, logger
from foundation.virtual import utility, stimulus, recording, fnn


@keys
class VisualRecordingCorrelation:
    """Visual Recording Correlation"""

    @property
    def keys(self):
        return [
            fnn.Model,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Correlation,
            utility.Burnin,
            utility.Bool.proj(perspective="bool"),
            utility.Bool.proj(modulation="bool"),
        ]

    @rowproperty
    def units(self):
        """
        Returns
        -------
        1D array
            [units] -- unitwise correlations
        """
        from foundation.recording.compute.visual import VisualTrials
        from foundation.utility.response import Correlation
        from foundation.stimulus.video import VideoSet
        from foundation.fnn.model import Model
        from foundation.fnn.data import Data
        from foundation.utils.response import Trials, concatenate
        from foundation.utils import cuda_enabled

        # load model
        model = (Model & self.item).model(device="cuda" if cuda_enabled() else "cpu")

        # load data
        data = (Data & self.item).link.compute

        # trial set
        trialset = {"trialset_id": data.trialset_id}

        # videos
        videos = (VideoSet & self.item).members
        videos = videos.fetch("KEY", order_by=videos.primary_key)

        # trials, targets, predictions
        trials = []
        targs = []
        preds = []

        with cache_rowproperty():

            for video in tqdm(videos, desc="Videos"):

                # trials
                trial_ids = (VisualTrials & trialset & video & self.item).trial_ids

                # no trials for video
                if not trial_ids:
                    logger.warning(f"No trials found for video_id `{video['video_id']}`")
                    continue

                # stimuli
                stimuli = data.trial_stimuli(trial_ids)

                # units
                units = data.trial_units(trial_ids)

                # perspectives
                if self.item["perspective"]:
                    perspectives = data.trial_perspectives(trial_ids)
                else:
                    perspectives = repeat(None)

                # modulations
                if self.item["modulation"]:
                    modulations = data.trial_modulations(trial_ids)
                else:
                    modulations = repeat(None)

                # video targets and predictions
                _targs = []
                _preds = []

                for s, p, m, u in zip(stimuli, perspectives, modulations, units):

                    # generate prediction
                    r = model.generate_response(stimuli=s, perspectives=p, modulations=m)
                    r = np.stack(list(r), axis=0)

                    _targs.append(u)
                    _preds.append(r)

                assert len(trial_ids) == len(_targs) == len(_preds)

                trials.append(trial_ids)
                targs.append(_targs)
                preds.append(_preds)

        # no trials at all
        if not trials:
            logger.warning(f"No trials found")
            return

        # correlations
        cc = (Correlation & self.item).link.correlation
        correlations = []

        for i in tqdm(range(data.units), desc="Units"):

            # unit targets and predictions
            unit_targ = []
            unit_pred = []

            for index, t, p in zip(trials, targs, preds):

                # target and prediction trials
                _unit_targ = Trials([_[:, i] for _ in t], index=index)
                _unit_pred = Trials([_[:, i] for _ in p], index=index)

                assert _unit_targ.matches(_unit_pred)

                unit_targ.append(_unit_targ)
                unit_pred.append(_unit_pred)

            # concatenated targets and predictions
            unit_targ = concatenate(*unit_targ, burnin=self.item["burnin"])
            unit_pred = concatenate(*unit_pred, burnin=self.item["burnin"])

            # unit correlation
            correlations.append(cc(unit_targ, unit_pred))

        return np.array(correlations)


@keys
class VisualResp:
    """Visual Mean Response Over Repeated Presentation and Corresponding Stimuli"""

    # @property
    # def keys(self):
    #     return [
    #         recording.ScanUnits,
    #         recording.TrialFilterSet,
    #         stimulus.VideoSet,
    #         utility.Resize,
    #         utility.Resolution,
    #         utility.Resample,
    #         utility.Offset,
    #         utility.Rate,
    #         utility.Burnin,
    #     ]

    @property
    def keys(self):
        return [
            fnn.Model,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Resize,
            utility.Burnin,
            utility.Bool.proj(perspective="bool"),
            utility.Bool.proj(modulation="bool"),
        ]

    @rowproperty
    def video_response(self):
        """
        Returns
        -------
            videos     : np.ndarray  [videos, time, height, width, channel]
            responses  : np.ndarray  [videos, time, units]
        """
        from foundation.recording.compute.visual import VisualTrials
        from foundation.utility.response import Correlation
        from foundation.stimulus.video import VideoSet
        from foundation.fnn.model import Model
        from foundation.fnn.data import Data
        from foundation.utils.response import Trials, concatenate
        from foundation.utils import cuda_enabled

        # load model
        model = (Model & self.item).model(device="cuda" if cuda_enabled() else "cpu")

        # load data
        data = (Data & self.item).link.compute

        # trial set
        trialset = {"trialset_id": data.trialset_id}

        # videos
        videos = (VideoSet & self.item).members
        videos = videos.fetch("KEY", order_by=videos.primary_key)

        # trials, targets, predictions
        trials = []
        targs = []
        preds = []


        with cache_rowproperty():

            for video in tqdm(videos, desc="Videos"):

                # trials
                trial_ids = (VisualTrials & trialset & video & self.item).trial_ids

                # no trials for video
                if not trial_ids:
                    logger.warning(f"No trials found for video_id `{video['video_id']}`")
                    continue

                # stimuli
                stimuli = data.trial_stimuli(trial_ids)

                # units
                units = data.trial_units(trial_ids)

                # perspectives
                if self.item["perspective"]:
                    perspectives = data.trial_perspectives(trial_ids)
                else:
                    perspectives = repeat(None)

                # modulations
                if self.item["modulation"]:
                    modulations = data.trial_modulations(trial_ids)
                else:
                    modulations = repeat(None)

                # video targets and predictions
                _targs = []
                _preds = []

                for s, p, m, u in zip(stimuli, perspectives, modulations, units):

                    # generate prediction
                    r = model.generate_response(stimuli=s, perspectives=p, modulations=m)
                    r = np.stack(list(r), axis=0)

                    _targs.append(u)
                    _preds.append(r)

                assert len(trial_ids) == len(_targs) == len(_preds)

                trials.append(trial_ids)
                targs.append(_targs)
                preds.append(_preds)

        # no trials at all
        if not trials:
            logger.warning(f"No trials found")
            return

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
        traceset = dict(traceset_id=(recording.ScanUnits & self.item).fetch1("traceset_id"))

        # videos
        video_ids = (VideoSet & self.item).members
        video_ids = video_ids.fetch("KEY", order_by=video_ids.primary_key)

        # unit order
        unit_order = (
            merge(recording.TraceSet.Member & traceset, recording.ScanUnitOrder & self.item)
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
                    logger.warning(f"No trials found for video_id `{video_id['video_id']}`")
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
                        ResizedVideo 
                        * recording.TrialVideo
                        * recording.ResampledTrial
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
                responses.append(trial_resps.mean(axis=0))  # [time, units]
                trial_video_index = np.stack(trial_video_index, axis=0) 
                assert np.all(trial_video_index == trial_video_index[0])
                trial_video_index = trial_video_index[0]
                resized_frames = (ResizedVideo & video_id & self.item).fetch1("video")
                frames.append(resized_frames[trial_video_index].astype(np.uint8))
        responses = np.stack(responses, axis=0)  # [videos, time, units]
        responses = responses[:, self.item["burnin"]:, :]  # [videos, time, units]
        frames = np.stack(frames, axis=0)[:, self.item["burnin"]:, :]  # [videos, time, height, width, channel]
        assert responses.shape[0] == frames.shape[0]
        return frames, responses
