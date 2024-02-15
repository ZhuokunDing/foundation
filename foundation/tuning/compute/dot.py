from djutils import keys, rowproperty, cache_rowproperty, merge
from foundation.virtual import recording, stimulus, utility, tuning
import numpy as np
import torch
import pandas as pd
from foundation.utils import tqdm, logger
from lmfit import Model

# ---------------------------- Dot ----------------------------

# -- DotResponse Interface --

class DotResponseType:
    """Tuning Dot"""

    @rowproperty
    def dots_responses(self):
        """
        Returns
        -------
            dots       : pd.DataFrame, rows are stimulus.compute.video.dot
            responses  : np.ndarray [dots, units]
        """
        raise NotImplementedError()

    @rowproperty
    def height(self):
        """
        Returns
        -------
        float
            height of the monitor in number of dots
        """
        raise NotImplementedError()
    
    @rowproperty
    def width(self):
        """
        Returns
        -------
        float
            width of the monitor in number of dots
        """
        raise NotImplementedError()
    

# -- Dot Types --

@keys
class RecordingDot(DotResponseType):
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
            utility.Offset.proj(tuning_offset_id="offset_id"),
        ]

    @rowproperty
    def dots_responses(self):
        from foundation.stimulus.video import Video, VideoSet
        from foundation.stimulus.compute.video import SquareDotType
        from foundation.recording.compute.visual import VisualTrials
        from foundation.recording.trace import TraceSet, Trace
        from foundation.recording import scan
        from foundation.utility.resample import Offset, Rate
        from foundation.utility.response import Burnin

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
        # sampling offset and rate
        period = (Rate & self.item).link.period  # in seconds
        tuning_offset = (Offset & dict(offset_id=self.item['tuning_offset_id'])).link.offset  # in seconds
        burnin = (Burnin & self.item).fetch1('burnin')  # in frames
        with cache_rowproperty():
            # parameter per dot
            dots = []  # dot parameters
            # visual responses
            responses = []  # [dots, trials]

            for video_id in tqdm(video_ids, desc="Videos"):
                # get trial ids when the video is shown
                trial_ids = (
                    VisualTrials & trialset & self.item & video_id
                ).trial_ids  # trials filtered by video and TrialFilterSet

                # no trials for video
                if not trial_ids:
                    logger.warning(
                        f"No trials found for video_id `{self.item['video_id']}`"
                    )
                    continue
                
                for trial_id in trial_ids:
                    # get video
                    video = (Video & video_id).link.compute
                    assert issubclass(
                        video.__class__, SquareDotType
                    ), "Video type is not SquareDotType"
                    # get parameters for each dot and their start and end times
                    dot_params, dot_starts, dot_ends = zip(*video.dots())
                    dots.extend(dot_params)
                    # trial responses
                    _trial_resps = (
                        recording.ResampledTraces
                        & dict(trial_id=trial_id)
                        & traceset
                        & self.item
                    ).fetch1("traces")  # [time, units], units are ordered by traceset_index
                    resampled_times = np.arange(len(_trial_resps)) * period - tuning_offset
                    assert len(resampled_times) > burnin, f"Burnin frames are more than a single trial!"
                    resampled_times[:burnin] = np.nan
                    assert len(resampled_times) == len(_trial_resps)
                    for i, (start, end) in enumerate(zip(dot_starts, dot_ends)):
                        mask = (resampled_times >= start) & (resampled_times < end)  # [time, traces]
                        responses.append(_trial_resps[mask].mean(axis=0))  # if no time points are in the mask, the mean will be nan
            responses = np.stack(responses, axis=0)[:, unit_order]  # [dots, traces], traces are ordered by unit_id
        dots = pd.DataFrame(dots)
        assert len(responses) == len(dots)
        assert dots.height.unique().size == 1, "All dots must have the same monitor height"
        assert dots.width.unique().size == 1, "All dots must have the same monitor width"
        return dots, responses


# ---------------------------- Dot STA ----------------------------

@keys
class DotSta():

    @property
    def keys(self):
        return [
            tuning.DotResponse,
            tuning.DotOnOff,
        ]
    
    @rowproperty
    def on(self):
        return (tuning.DotOnOff & self.item).fetch1("on")

    @staticmethod    
    def render_stimuli(dot):
        stim = np.zeros((dot.height, dot.width), dtype=np.float32)
        stim[dot.y, dot.x] = 1
        return stim

    @staticmethod
    def compute_sta(stimuli, response):
        """
        Args:
            stimuli: [trials, height, width]
            response: [trials, traces]
        Returns:
            sta: [height, width, traces] as cross correlation between stimuli and response
        """
        x = torch.tensor(stimuli, dtype=torch.float32, requires_grad=False, device="cuda")
        y = torch.tensor(response, dtype=torch.float32, requires_grad=False, device="cuda")
        # standardize
        x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
        y = (y - y.mean(dim=0, keepdim=True)) / y.std(dim=0, keepdim=True)
        # fill nan with 0
        x[torch.isnan(x)] = 0
        y[torch.isnan(y)] = 0
        # compute sta
        sta = torch.einsum("thw,tu->hwu", x, y) / x.shape[0]
        # predict response
        y_hat = torch.einsum("thw,hwu->tu", x, sta)
        y_hat = (y_hat - y_hat.mean(dim=0, keepdim=True)) / y_hat.std(dim=0, keepdim=True)
        # compute correlation
        corr = torch.einsum("tu,tu->u", y, y_hat) / y.shape[0]
        sta = sta.cpu().numpy()
        corr = corr.cpu().numpy()
        return sta, corr

    @rowproperty
    def dots_responses(self):
        from foundation.tuning import dot
        dots, responses = (dot.DotResponse & self.item).link.compute.dots_responses
        responses = responses[dots.on == self.on]
        dots = dots[dots.on == self.on]
        return dots, responses

    @rowproperty
    def sta(self):
        from foundation.tuning import dot
        from itertools import product
        dots, responses = self.dots_responses
        # render stimuli
        dots['stimuli'] = dots.apply(dot.DotSta.render_stimuli, axis=1)
        dot_coverage = np.zeros((self.height, self.width), dtype=bool)
        for x, y, o in product(dots.x.unique(), dots.y.unique(), dots.on.unique()):
            if not ((dots.x == x) & (dots.y == y) & (dots.on == o)).any():
                continue
            dot_coverage[y, x] = True
        # compute sta
        sta, pred_corr = dot.DotSta.compute_sta(dots.stimuli.to_numpy(), responses)
        return sta, pred_corr, dot_coverage

    @staticmethod
    def gaussian2d(x, y, mu_x, mu_y, theta, sigma_x, sigma_y, amp, offset):
        _x = x - mu_x
        _y = y - mu_y

        cos = np.cos(theta)
        sin = np.sin(theta)

        v_x = cos * _x - sin * _y
        v_y = sin * _x + cos * _y

        s_x = (v_x / sigma_x) ** 2
        s_y = (v_y / sigma_y) ** 2

        return amp * np.exp(-0.5 * (s_x + s_y)) + offset

    def fit_gaussian2d(self,sta, dot_coverage, max_rf_size=None):
        y, x = np.where(dot_coverage)
        z = sta[dot_coverage]
        x, y, z = x.ravel(), y.ravel(), z.ravel()
        amp = z.max() - z.min()
        offset = z.min()
        mu_x = x[z.argmax()]
        mu_y = y[z.argmax()]
        var_x = (z * (x - mu_x) ** 2).sum() / z.sum()
        var_x = np.clip(var_x, 0, None)
        var_y = (z * (y - mu_y) ** 2).sum() / z.sum()
        var_y = np.clip(var_y, 0, None)  
        max_rf_size = max_rf_size or np.max([x.max(), y.max()])/2
        model = Model(self.gaussian2d, independent_vars=["x", "y"])
        params = model.make_params(
            mu_x=dict(value=mu_x, min=x.min(), max=x.max()),
            mu_y=dict(value=mu_y, min=y.min(), max=y.max()),
            theta=dict(value=0, min=0, max=np.pi),
            sigma_x=dict(value=max_rf_size / 2, min=0, max=max_rf_size),
            sigma_y=dict(value=max_rf_size / 2, min=0, max=max_rf_size),
            amp=dict(value=amp, min=0, max=1),
            offset=dict(value=offset, min=-1, max=1)
        )
        fit_result = model.fit(z, params, x=x, y=y)
        fitted_sta = np.full(dot_coverage.shape, np.nan)
        fitted_sta[y, x] = fit_result.eval(x=x, y=y)
        fitted_sta[np.isnan(fitted_sta)] = 0
        return fit_result, fitted_sta

    @rowproperty
    def sta_gaussianed_fit(self):
        sta, dot_coverage = (tuning.DotSta & self.item).fetch1("sta", "dot_coverage")
        fit_result, fitted_sta = self.fit_gaussian2d(sta, dot_coverage)
        params = fit_result.best_values
        height, width = fitted_sta.shape
        # convert mu_x and mu_y to [-1, 1]
        params['mu_x'] = params['mu_x'].value / width * 2 - 1
        params['mu_y'] = params['mu_y'].value / height * 2 - 1
        # compute predicted responses and prediction correlation
        dots, responses = self.dots_responses
        stim = dots.apply(DotSta.render_stimuli, axis=1).to_numpy()
        stim = torch.tensor(np.stack(stim, axis=0), dtype=torch.float32) 
        responses = torch.tensor(responses, dtype=torch.float32)
        stim = (stim - stim.mean(dim=0, keepdim=True)) / stim.std(dim=0, keepdim=True)
        pred_responses = torch.einsum("thw,hwu->tu", stim, torch.tensor(fitted_sta, dtype=torch.float32))
        pred_responses = (pred_responses - pred_responses.mean(dim=0, keepdim=True)) / pred_responses.std(dim=0, keepdim=True)
        responses = (responses - responses.mean(dim=0, keepdim=True)) / responses.std(dim=0, keepdim=True)
        pred_corr = torch.einsum("tu,tu->u", responses, pred_responses) / responses.shape[0]
        pred_corr = pred_corr.cpu().numpy()
        return dict(
            **params,
            fit_pred_corr=pred_corr
        )
        

