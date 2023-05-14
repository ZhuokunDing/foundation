import numpy as np
import pandas as pd
from djutils import keys, merge, rowproperty, rowmethod
from foundation.fnn.data import DataSet, DataSetComponents, DataSpec


@keys
class PreprocessedData:
    """Load Preprocessed Data"""

    @property
    def key_list(self):
        return [
            DataSet,
            DataSpec.Preprocess,
        ]

    @rowproperty
    def trials(self):
        from foundation.recording.trial import Trial, TrialSet

        key = merge(self.key, DataSetComponents)
        return Trial & (TrialSet & key).members

    @rowproperty
    def trial_samples(self):
        from foundation.recording.trial import TrialSamples

        key = merge(self.key, DataSpec.Preprocess)

        trials = merge(self.trials, TrialSamples & key)
        trial_id, samples = trials.fetch("trial_id", "samples", order_by="trial_id", limit=5)  # TODO

        return pd.Series(data=samples, index=pd.Index(trial_id, name="trial_id"))

    @rowproperty
    def trial_video(self):
        from foundation.recording.trial import TrialVideo
        from foundation.recording.cache import ResampledVideo
        from foundation.stimulus.cache import ResizedVideo
        from fnn.data import NpyFile

        key = merge(self.key, DataSpec.Preprocess)

        trials = merge(self.trials, TrialVideo, ResizedVideo & key, ResampledVideo & key)
        trial_id, video, index = trials.fetch("trial_id", "video", "index", order_by="trial_id", limit=5)  # TODO

        data = [NpyFile(v, indexmap=np.load(i)) for v, i in zip(video, index)]
        return pd.Series(data=data, index=pd.Index(trial_id, name="trial_id"))

    @rowmethod
    def trial_traces(self, suffix="p"):
        from foundation.recording.compute import StandardizeTraces
        from foundation.recording.cache import ResampledTraces
        from fnn.data import NpyFile

        if suffix not in ["p", "m", "u"]:
            raise ValueError("Suffix must be one of {'p', 'm', 'u'}")

        proj = {f"{k}_id": f"{k}_id_{suffix}" for k in ["traceset", "offset", "resample", "standardize"]}
        key = merge(self.key, DataSetComponents, DataSpec.Preprocess).proj(..., **proj)

        transform = (StandardizeTraces & key).transform

        trials = merge(self.trials, ResampledTraces & key & "finite")
        trial_id, traces = trials.fetch("trial_id", "traces", order_by="trial_id", limit=5)  # TODO

        data = [NpyFile(t, transform=transform) for t in traces]
        return pd.Series(data=data, index=pd.Index(trial_id, name="trial_id"))

    @rowproperty
    def visual_dataset(self):
        from fnn.data import Dataset

        data = [
            self.trial_samples.rename("samples"),
            self.trial_video.rename("stimuli"),
            self.trial_traces("p").rename("perspectives"),
            self.trial_traces("m").rename("modulations"),
            self.trial_traces("u").rename("units"),
        ]
        df = pd.concat(data, axis=1, join="outer")
        assert not df.isnull().values.any()

        return Dataset(df)
