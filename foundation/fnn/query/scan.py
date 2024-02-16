from djutils import keys, merge
from foundation.virtual import utility, stimulus, scan, recording, fnn
import pandas as pd

@keys
class VisualScanRecording:
    """Visual Scan Recording"""

    @property
    def keys(self):
        return [
            (scan.Scan * fnn.Data) & fnn.Data.VisualScan,
        ]

    @property
    def units(self):
        key = self.key.fetch("KEY")
        units = fnn.Data.VisualScan * recording.ScanUnitOrder * recording.Trace.ScanUnit * recording.ScanUnit & key
        return units.proj("trace_order")


@keys
class VisualScanCorrelation:
    """Visual Scan Correlation"""

    @property
    def keys(self):
        return [
            (scan.Scan * fnn.Model) & fnn.Data.VisualScan,
            recording.TrialFilterSet,
            stimulus.VideoSet,
            utility.Burnin,
            utility.Bool.proj(perspective="bool"),
            utility.Bool.proj(modulation="bool"),
        ]

    def cc_norm(self):
        data_scan_spec = fnn.Data.VisualScan.proj(
            "spec_id",
            "trace_filterset_id",
            "pipe_version",
            "animal_id",
            "session",
            "scan_idx",
            "segmentation_method",
            "spike_method",
        ) * fnn.Spec.VisualSpec.proj(
            "rate_id", offset_id="offset_id_unit", resample_id="resample_id_unit"
        )
        all_unit_trace_rel = (
            self.key
            * data_scan_spec  # data_id -> specs + scan key
            * recording.ScanUnitOrder  # scan key + trace_filterset_id -> trace_ids
            * recording.Trace.ScanUnit  # trace_id -> unit key
        )
        all_units_df = all_unit_trace_rel.fetch(format="frame").reset_index()
        # fetch cc_max
        cc_max = (
            (recording.VisualMeasure & utility.Measure.CCMax & all_unit_trace_rel)
            .fetch(format="frame")
            .reset_index()
            .rename(columns={"measure": "cc_max"})
        )
        # fetch cc_abs
        cc_abs_df = pd.DataFrame(
            (
                (
                    fnn.VisualRecordingCorrelation 
                    & utility.Correlation.CCSignal
                ).proj(
                    ..., trace_order="unit"
                )
                & all_unit_trace_rel
            )
            .fetch(as_dict=True)  # this fetch is very slow
        ).reset_index().rename(columns={"correlation": "cc_abs"})
        # compute cc_norm
        cc_norm_df = (
            all_units_df.merge(cc_abs_df, how="left", validate="one_to_one")
            .merge(cc_max, how="left", validate="one_to_one")
            .assign(cc_norm=lambda df: df.cc_abs / df.cc_max)
        )
        return cc_norm_df
