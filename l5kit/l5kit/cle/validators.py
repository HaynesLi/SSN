from abc import abstractmethod
from collections import defaultdict
from enum import IntEnum
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Type

import torch
from typing_extensions import Protocol

from l5kit.cle import metrics
from l5kit.simulation.unroll import SimulationOutputCLE, TrajectoryStateIndices


class ValidatorOutput(NamedTuple):

    is_valid_scene: bool
    failed_frames: List[int]


class SupportsMetricValidate(Protocol):

    validator_name: str
    requires_metric: List[str]

    @abstractmethod
    def validate(self, metric_results: Dict[str, torch.Tensor],
                 simulation_output: SimulationOutputCLE) -> ValidatorOutput:
        raise NotImplementedError


class DurationMode(IntEnum):

    TOTAL = 0
    CONTINUOUS = 1


class RangeValidator(SupportsMetricValidate):


    def __init__(self, validator_name: str, metric: Type[metrics.SupportsMetricCompute],
                 min_value: Optional[float] = None, max_value: Optional[float] = None,
                 violation_duration_s: float = 0.0, duration_mode: DurationMode = DurationMode.TOTAL):
        # Requires at least one value specification
        if min_value is None and max_value is None:
            raise ValueError("At least one parameter must be "
                             "specified: min_value or max_value.")

        if min_value is not None and max_value is not None:
            if min_value >= max_value:
                raise ValueError("Minimum value cannot be greater or equal"
                                 " to the maximum value.")

        self.validator_name = validator_name
        self.metric_name = metric.metric_name
        self.requires_metric = [self.metric_name]
        self.min_value = min_value
        self.max_value = max_value
        self.validation_duration_s = violation_duration_s
        self.duration_mode = duration_mode

    @staticmethod
    def cumsum_with_reset(timestamp_diff: torch.Tensor,
                          validation_mask: torch.Tensor) -> torch.Tensor:

        cumsum = torch.zeros_like(timestamp_diff)
        accumulator = 0.0
        for idx, (ts, vmask) in enumerate(zip(timestamp_diff, validation_mask)):
            if not vmask:
                accumulator = 0.0
            else:
                accumulator += ts.item()
            cumsum[idx] = accumulator
        return cumsum

    def validate(self, metric_results: Dict[str, torch.Tensor],
                 simulation_output: SimulationOutputCLE) -> ValidatorOutput:

        result = metric_results[self.metric_name]
        validation_mask = torch.zeros_like(result, dtype=torch.bool)

        if self.min_value is not None:
            validation_mask |= result < self.min_value

        if self.max_value is not None:
            validation_mask |= result > self.max_value

        # Immediate failure if there is a violation and the allowed
        # duration is zero
        if self.validation_duration_s <= 0.0:
            # Log the failed frames into the scene tracker
            failed_frame_indexes = torch.nonzero(validation_mask).squeeze(1)

            failed_frame_indexes = failed_frame_indexes.cpu().numpy().tolist()

            is_valid_scene = len(failed_frame_indexes) == 0
            return ValidatorOutput(is_valid_scene, failed_frame_indexes)

        # If duration is greater than zero, then we check
        # if there was a violation greater than the
        # allowed duration
        ego_states = simulation_output.simulated_ego_states
        timestamps = ego_states[:, TrajectoryStateIndices.TIME.value]

        # Diff of the timestamps
        pad = torch.as_tensor([0], device=timestamps.device)
        pad_ts = torch.cat((pad, timestamps))
        ts_diff = pad_ts[1:] - pad_ts[:-1]

        # Total mode: we sum all violation durations
        if self.duration_mode == DurationMode.TOTAL:
            # Build a cumulative sum (masked by the validation mask)
            ts_valid_cumsum = (ts_diff * validation_mask).cumsum(dim=0)
            ts_valid_cumsum = ts_valid_cumsum * validation_mask

        # Continuous mode: we check if any of the durations
        # violated the constraint
        if self.duration_mode == DurationMode.CONTINUOUS:
            # Cumulative sum here is computed with reset, if there is a valid
            # metric computation between two consecutive chunks of invalid
            # metrics, then this valid frame will reset the cumulative sum
            # of the timestamp diffs
            ts_valid_cumsum = RangeValidator.cumsum_with_reset(ts_diff, validation_mask)

        # Check which timestamps violated the duration
        ts_cumsum_violated = ts_valid_cumsum > self.validation_duration_s

        # Get the frame indexes and track them
        violation_indexes = torch.nonzero(ts_cumsum_violated).squeeze(1)
        violation_indexes = violation_indexes.cpu().numpy().tolist()

        is_valid_scene = len(violation_indexes) == 0
        return ValidatorOutput(is_valid_scene, violation_indexes)


class SupportsValidationAggregation(Protocol):


    def aggregate(self, scene_validation_results:
                  Dict[int, Dict[str, ValidatorOutput]]) -> Dict[str, Any]:

        raise NotImplementedError


class ValidationCountingAggregator(SupportsValidationAggregation):

    def __init__(self, failed_frames: bool = False):
        self.failed_frames = failed_frames

    def aggregate_scenes(self, scene_validation_results:
                         Dict[int, Dict[str, ValidatorOutput]]) -> Dict[str, Any]:
        aggregation: DefaultDict[str, int] = defaultdict(int)
        for _, validator_dict in scene_validation_results.items():
            for validator_name, validator_output in validator_dict.items():
                # Aggregate the number of failed frames in the scene
                if self.failed_frames:
                    aggregation[validator_name] += len(validator_output.failed_frames)/100
                else:  # or the number of scenes
                    aggregation[validator_name] += (not validator_output.is_valid_scene)/100
        aggregation_torch = {k: torch.as_tensor(v) for k, v in aggregation.items()}
        # aggregation_torch = {k: int(v) for k, v in aggregation}
        return aggregation_torch

    def aggregate(self, scene_validation_results:
                  Dict[int, Dict[str, ValidatorOutput]]) -> Dict[str, Any]:

        agg_scenes = self.aggregate_scenes(scene_validation_results)
        return agg_scenes


class FailedFrame(NamedTuple):

    scene_id: int
    frame_index: int


class ValidationFailedFramesAggregator:
    """This aggregator will aggregate all failed frames (and scenes)."""

    def aggregate_scenes(self, scene_validation_results:
                         Dict[int, Dict[str, ValidatorOutput]]) -> Dict[str, Any]:

        aggregation: DefaultDict[str, List[FailedFrame]] = defaultdict(list)

        for scene_id, validator_dict in scene_validation_results.items():
            for validator_name, validator_output in validator_dict.items():
                if len(validator_output.failed_frames) > 0:
                    failed_fames = [FailedFrame(scene_id, frame_index)
                                    for frame_index in validator_output.failed_frames]
                    aggregation[validator_name].extend(failed_fames)

        aggregation_torch = {k: torch.as_tensor(v) for k, v in aggregation.items()}
        return aggregation_torch

    def aggregate(self, scene_validation_results:
                  Dict[int, Dict[str, ValidatorOutput]]) -> Dict[str, Any]:
        agg_scenes = self.aggregate_scenes(scene_validation_results)
        return agg_scenes
