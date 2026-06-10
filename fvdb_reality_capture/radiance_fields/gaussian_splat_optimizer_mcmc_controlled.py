# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from fvdb import GaussianSplat3d

from fvdb_reality_capture.radiance_fields.base_gaussian_splat_optimizer import (
    BaseGaussianSplatOptimizer,
    splat_optimizer,
)
from fvdb_reality_capture.radiance_fields.gaussian_splat_optimizer import (
    GaussianSplatOptimizer,
    GaussianSplatOptimizerConfig,
)
from fvdb_reality_capture.sfm_scene.sfm_scene import SfmScene


class MetricController:
    """
    A feedback controller that adjusts the Gaussian insertion rate based on
    the trend of an observed quality metric (PSNR or SSIM).

    The controller maintains a sliding window of (step, metric_value, gaussian_count)
    observations and uses two complementary signals for mode transitions:

    **Dual-signal control:**

    - **INSERT -> HOLD** (effectiveness gating): During insertion, the controller
      tracks the marginal quality improvement per Gaussian added
      (dMetric/dGaussians).  When this effectiveness decays to a configurable
      fraction of its peak value, the controller switches to HOLD.  This
      automatically detects diminishing returns and plateaus the Gaussian count.

    - **HOLD -> INSERT** (stall detection): During holding, the controller
      monitors the quality-vs-step slope.  When improvement stalls, it resumes
      insertion to add more capacity.

    This design defaults to INSERT after warmup (like the standard MCMC mode),
    giving aggressive early growth, while the effectiveness gate automatically
    detects when the model has enough Gaussians and stops.  The stall detector
    provides a recovery mechanism if quality later plateaus.

    When ``effectiveness_decay_ratio`` is ``0.0`` (disabled), the controller
    falls back to the original slope-based logic for INSERT -> HOLD transitions,
    preserving full backward compatibility.

    Additional features:
        - **Hysteresis** on slope thresholds to prevent chattering.
        - **Anti-windup**: burst-level effectiveness check that can permanently
          suppress insertion if a burst is unproductive.
        - **Cooldown**: forced HOLD period after each INSERT -> HOLD transition.

    Args:
        trend_window: Number of recent measurements to use for linear fits.
        slope_threshold: Slope (dMetric/dStep) below which HOLD->INSERT triggers.
        slope_threshold_high: Slope above which INSERT->HOLD triggers (only used
            when ``effectiveness_decay_ratio == 0.0``).  Must be >= ``slope_threshold``.
        max_insertion_rate: Insertion rate to use when inserting.
        min_measurements_before_control: Minimum observations before the controller
            activates.  During warmup the controller returns ``max_insertion_rate``.
        effectiveness_threshold: Minimum dMetric / dGaussians required for an
            insertion burst to be considered effective.  Set to ``0.0`` to disable
            anti-windup.
        cooldown_probes: Minimum number of probe intervals to remain in HOLD
            after an INSERT->HOLD transition.
        effectiveness_decay_ratio: Fraction of peak effectiveness below which
            the controller transitions from INSERT to HOLD.  Set to ``0.0`` to
            disable effectiveness gating and use the original slope-based logic.
    """

    def __init__(
        self,
        trend_window: int = 5,
        slope_threshold: float = 1e-4,
        slope_threshold_high: float = 2e-4,
        max_insertion_rate: float = 1.05,
        min_measurements_before_control: int = 3,
        effectiveness_threshold: float = 0.0,
        cooldown_probes: int = 0,
        effectiveness_decay_ratio: float = 0.0,
    ):
        if trend_window < 2:
            raise ValueError("trend_window must be >= 2 for a meaningful linear fit")
        if slope_threshold_high < slope_threshold:
            raise ValueError(
                f"slope_threshold_high ({slope_threshold_high}) must be >= slope_threshold ({slope_threshold})"
            )
        if cooldown_probes < 0:
            raise ValueError(f"cooldown_probes must be >= 0, got {cooldown_probes}")
        if not (0.0 <= effectiveness_decay_ratio < 1.0):
            raise ValueError(f"effectiveness_decay_ratio must be in [0.0, 1.0), got {effectiveness_decay_ratio}")
        self._trend_window = trend_window
        self._slope_threshold = slope_threshold
        self._slope_threshold_high = slope_threshold_high
        self._max_insertion_rate = max_insertion_rate
        self._min_measurements = min_measurements_before_control
        self._effectiveness_threshold = effectiveness_threshold
        self._cooldown_probes = cooldown_probes
        self._effectiveness_decay_ratio = effectiveness_decay_ratio

        # History of (step, metric_value, gaussian_count) triples
        self._history: list[tuple[int, float, int]] = []

        # Cached state
        self._current_insertion_rate: float = max_insertion_rate
        self._current_slope: float | None = None

        # Anti-windup: burst tracking
        self._burst_start_metric: float | None = None
        self._burst_start_gaussians: int | None = None
        self._saturated: bool = False

        # Post-burst cooldown counter (probes remaining before INSERT is allowed)
        self._cooldown_remaining: int = 0

        # Effectiveness gating state: tracks dMetric/dGaussians during INSERT phases
        self._insert_phase_start_idx: int | None = None
        self._peak_effectiveness: float = 0.0

        # Baseline optimization slope (dMetric/dStep from the last HOLD phase).
        # Subtracted during INSERT to isolate the densification contribution.
        self._baseline_optimization_slope: float = 0.0

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    @property
    def history(self) -> list[tuple[int, float, int]]:
        """Return the full observation history as (step, metric_value, gaussian_count) triples."""
        return list(self._history)

    @property
    def current_slope(self) -> float | None:
        """Return the most recently computed slope, or None if not yet available."""
        return self._current_slope

    @property
    def peak_effectiveness(self) -> float:
        """Return the peak dMetric/dGaussians observed during INSERT phases."""
        return self._peak_effectiveness

    @property
    def current_insertion_rate(self) -> float:
        """Return the current insertion rate as determined by the controller."""
        return self._current_insertion_rate

    @property
    def is_inserting(self) -> bool:
        """Return True if the controller is currently commanding insertion (rate > 1.0)."""
        return self._current_insertion_rate > 1.0

    @property
    def is_saturated(self) -> bool:
        """Return True if the controller has detected ineffective insertion and suppressed it."""
        return self._saturated

    @property
    def is_cooling_down(self) -> bool:
        """Return True if the controller is in post-burst cooldown (forced HOLD)."""
        return self._cooldown_remaining > 0

    def observe(self, step: int, metric_value: float, gaussian_count: int = 0) -> float:
        """
        Record a new metric observation and update the insertion rate.

        Args:
            step: The training step at which the metric was measured.
            metric_value: The quality metric value (PSNR in dB, or SSIM in 0-1).
            gaussian_count: Current number of Gaussians in the model (used for
                effectiveness tracking and anti-windup).

        Returns:
            The updated insertion rate.
        """
        was_inserting = self.is_inserting

        self._history.append((step, metric_value, gaussian_count))

        if len(self._history) < self._min_measurements:
            self._current_slope = None
            self._current_insertion_rate = self._max_insertion_rate
            if self._burst_start_metric is None:
                self._burst_start_metric = metric_value
                self._burst_start_gaussians = gaussian_count
            if self._insert_phase_start_idx is None:
                self._insert_phase_start_idx = 0
            self._logger.debug(
                f"MetricController: observation {len(self._history)}/{self._min_measurements} "
                f"(warmup), step={step}, metric={metric_value:.4f}, rate={self._current_insertion_rate:.4f}"
            )
            return self._current_insertion_rate

        # Compute quality-vs-step slope for stall detection (HOLD->INSERT signal)
        window = self._history[-self._trend_window :]
        slope = self._linear_regression_slope([(s, m) for s, m, _g in window])
        self._current_slope = slope

        # --- INSERT->HOLD decision ---
        if was_inserting:
            if self._effectiveness_decay_ratio > 0.0:
                should_insert = not self._effectiveness_decayed(gaussian_count)
            else:
                should_insert = slope < self._slope_threshold_high
        else:
            # HOLD->INSERT: use quality-vs-step stall detection
            should_insert = slope < self._slope_threshold

        # Anti-windup: if saturated, override to HOLD
        if self._saturated:
            should_insert = False

        # Post-burst cooldown: suppress INSERT until the cooldown expires
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            should_insert = False
            self._logger.debug(
                f"MetricController: cooldown active ({self._cooldown_remaining} probes remaining), "
                f"forcing HOLD at step={step}"
            )

        # Track insertion burst transitions
        if not was_inserting and should_insert:
            # HOLD -> INSERT transition
            self._burst_start_metric = metric_value
            self._burst_start_gaussians = gaussian_count
            self._insert_phase_start_idx = len(self._history) - 1
            self._baseline_optimization_slope = slope
            self._peak_effectiveness = 0.0
            self._logger.info(
                f"MetricController: HOLD -> INSERT at step={step}, "
                f"metric={metric_value:.6f}, slope={slope:.2e}, gaussians={gaussian_count:,}, "
                f"baseline_opt_slope={slope:.2e}"
            )

        if was_inserting and not should_insert and self._burst_start_metric is not None:
            # INSERT -> HOLD transition: evaluate burst effectiveness
            delta_metric = metric_value - self._burst_start_metric
            delta_gaussians = gaussian_count - (self._burst_start_gaussians or 0)

            if delta_gaussians > 0 and self._effectiveness_threshold > 0.0:
                effectiveness = delta_metric / delta_gaussians
                if effectiveness < self._effectiveness_threshold:
                    self._saturated = True
                    self._logger.info(
                        f"MetricController: SATURATED at step={step}. "
                        f"Burst added {delta_gaussians:,} gaussians for {delta_metric:+.6f} quality "
                        f"(effectiveness={effectiveness:.2e}, threshold={self._effectiveness_threshold:.2e}). "
                        f"Suppressing future insertion."
                    )
                else:
                    self._logger.info(
                        f"MetricController: INSERT -> HOLD at step={step}. "
                        f"Burst added {delta_gaussians:,} gaussians for {delta_metric:+.6f} quality "
                        f"(effectiveness={effectiveness:.2e})"
                    )
            else:
                self._logger.info(
                    f"MetricController: INSERT -> HOLD at step={step}, "
                    f"delta_metric={delta_metric:+.6f}, delta_gaussians={delta_gaussians:,}"
                )

            self._burst_start_metric = None
            self._burst_start_gaussians = None
            self._insert_phase_start_idx = None

            if self._cooldown_probes > 0:
                self._cooldown_remaining = self._cooldown_probes
                self._logger.info(
                    f"MetricController: cooldown started for {self._cooldown_probes} probes at step={step}"
                )

        # Apply the control decision
        if should_insert:
            self._current_insertion_rate = self._max_insertion_rate
        else:
            self._current_insertion_rate = 1.0

        self._logger.debug(
            f"MetricController: step={step}, metric={metric_value:.4f}, "
            f"slope={slope:.6f}, threshold_lo={self._slope_threshold:.6f}, "
            f"threshold_hi={self._slope_threshold_high:.6f}, "
            f"rate={self._current_insertion_rate:.4f}, saturated={self._saturated}, "
            f"cooldown={self._cooldown_remaining}, peak_eff={self._peak_effectiveness:.2e}, "
            f"baseline_opt_slope={self._baseline_optimization_slope:.2e}"
        )

        return self._current_insertion_rate

    def _effectiveness_decayed(self, current_gaussian_count: int) -> bool:
        """
        Check whether insertion effectiveness has decayed enough to trigger HOLD.

        Computes dMetric/dGaussians over the INSERT-phase observations using
        linear regression after subtracting the optimization-only baseline
        (dMetric/dStep measured during the prior HOLD phase).  This isolates the
        densification contribution from the normal optimization improvement that
        would occur even without adding Gaussians.

        Updates the peak effectiveness and returns True if the current
        effectiveness has fallen below ``effectiveness_decay_ratio`` of the peak.

        Returns False (keep inserting) when there is insufficient data.
        """
        if self._insert_phase_start_idx is None:
            return False

        phase_points = self._history[self._insert_phase_start_idx :]
        if len(phase_points) < 2:
            return False

        g_start = phase_points[0][2]
        g_end = phase_points[-1][2]
        if g_end <= g_start:
            return False

        step_0 = phase_points[0][0]
        adjusted_points = [
            (step, metric - self._baseline_optimization_slope * (step - step_0), gc)
            for step, metric, gc in phase_points
        ]

        eff = self._effectiveness_slope(adjusted_points)

        if eff > self._peak_effectiveness:
            self._peak_effectiveness = eff
            self._logger.debug(
                f"MetricController: new peak effectiveness={eff:.2e} "
                f"(gaussians {g_start:,} -> {g_end:,}, baseline_slope={self._baseline_optimization_slope:.2e})"
            )

        if self._peak_effectiveness <= 0.0:
            return False

        ratio = eff / self._peak_effectiveness
        decayed = ratio < self._effectiveness_decay_ratio

        if decayed:
            self._logger.info(
                f"MetricController: INSERT -> HOLD (effectiveness decayed). "
                f"net dMetric/dGaussians: current={eff:.2e}, peak={self._peak_effectiveness:.2e}, "
                f"ratio={ratio:.3f} < threshold={self._effectiveness_decay_ratio:.3f}, "
                f"baseline_slope={self._baseline_optimization_slope:.2e}"
            )

        return decayed

    def get_insertion_rate(self) -> float:
        """Return the current insertion rate without recording a new observation."""
        return self._current_insertion_rate

    def state_dict(self) -> dict[str, Any]:
        """Serialize the controller state for checkpointing."""
        return {
            "trend_window": self._trend_window,
            "slope_threshold": self._slope_threshold,
            "slope_threshold_high": self._slope_threshold_high,
            "max_insertion_rate": self._max_insertion_rate,
            "min_measurements": self._min_measurements,
            "effectiveness_threshold": self._effectiveness_threshold,
            "effectiveness_decay_ratio": self._effectiveness_decay_ratio,
            "history": self._history,
            "current_insertion_rate": self._current_insertion_rate,
            "current_slope": self._current_slope,
            "burst_start_metric": self._burst_start_metric,
            "burst_start_gaussians": self._burst_start_gaussians,
            "saturated": self._saturated,
            "cooldown_probes": self._cooldown_probes,
            "cooldown_remaining": self._cooldown_remaining,
            "insert_phase_start_idx": self._insert_phase_start_idx,
            "peak_effectiveness": self._peak_effectiveness,
            "baseline_optimization_slope": self._baseline_optimization_slope,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the controller state from a checkpoint."""
        self._trend_window = state_dict["trend_window"]
        self._slope_threshold = state_dict["slope_threshold"]
        self._slope_threshold_high = state_dict.get("slope_threshold_high", self._slope_threshold)
        self._max_insertion_rate = state_dict["max_insertion_rate"]
        self._min_measurements = state_dict["min_measurements"]
        self._effectiveness_threshold = state_dict.get("effectiveness_threshold", 0.0)
        self._effectiveness_decay_ratio = state_dict.get("effectiveness_decay_ratio", 0.0)
        raw_history = state_dict["history"]
        # Migrate legacy 2-tuple history to 3-tuples (gaussian_count=0 for old entries)
        if raw_history and len(raw_history[0]) == 2:
            self._history = [(s, m, 0) for s, m in raw_history]
        else:
            self._history = raw_history
        self._current_insertion_rate = state_dict["current_insertion_rate"]
        self._current_slope = state_dict["current_slope"]
        # Accept both new key ("burst_start_metric") and legacy key ("burst_start_psnr")
        self._burst_start_metric = state_dict.get("burst_start_metric", state_dict.get("burst_start_psnr"))
        self._burst_start_gaussians = state_dict.get("burst_start_gaussians")
        self._saturated = state_dict.get("saturated", False)
        self._cooldown_probes = state_dict.get("cooldown_probes", 0)
        self._cooldown_remaining = state_dict.get("cooldown_remaining", 0)
        self._insert_phase_start_idx = state_dict.get("insert_phase_start_idx")
        self._peak_effectiveness = state_dict.get("peak_effectiveness", 0.0)
        self._baseline_optimization_slope = state_dict.get("baseline_optimization_slope", 0.0)

    @staticmethod
    def _linear_regression_slope(points: list[tuple[float, float]]) -> float:
        """
        Compute the slope of a least-squares linear fit to a list of (x, y) points.

        Uses the closed-form formula:
            slope = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)

        Args:
            points: A list of (x, y) tuples (only the first two elements are used).

        Returns:
            The slope of the best-fit line.
        """
        n = len(points)
        if n < 2:
            return 0.0

        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)
        sum_x2 = sum(p[0] * p[0] for p in points)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-15:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denom

    @classmethod
    def _effectiveness_slope(cls, history_points: list[tuple[int, float, int]]) -> float:
        """
        Compute dMetric/dGaussians from a slice of history triples.

        Extracts (gaussian_count, metric_value) pairs and fits a line.

        Args:
            history_points: A list of (step, metric_value, gaussian_count) triples.

        Returns:
            The slope of metric vs gaussian_count (dMetric/dGaussians).
        """
        xy = [(float(gc), mv) for _s, mv, gc in history_points]
        return cls._linear_regression_slope(xy)


class ExtremumSeekingController:
    """
    Extremum-seeking controller (ESC) for automatic Gaussian budget discovery.

    Uses discrete perturbation bursts (INSERT or PRUNE) followed by dwell periods
    to estimate the gradient dQuality/dGaussianCount.  The sign of this gradient
    drives action selection: positive -> INSERT, negative -> PRUNE, near-zero -> HOLD.

    State machine::

        WARMUP -> INSERT -> DWELL -> ESTIMATE -> INSERT | HOLD | PRUNE
                                                  HOLD -> ESTIMATE (after recheck)
                                                  PRUNE -> DWELL -> ESTIMATE -> ...

    All sizing parameters are self-normalizing (fractions of current count), so no
    scene-specific absolute thresholds are needed.

    Args:
        perturbation_fraction: Fraction of current Gaussian count to add/remove per burst.
        dwell_probes: Number of probe intervals to wait after a perturbation before
            estimating the gradient.
        gradient_window: Number of observations for gradient estimation via linear regression.
        min_warmup_probes: Warmup observations before control activates.
        recheck_probes: Probes between gradient re-estimates during HOLD.
        noise_deadband: Gradient magnitudes below this fraction of peak observed
            |gradient| are treated as near-zero (HOLD).
    """

    STATES = ("WARMUP", "INSERT", "DWELL", "ESTIMATE", "HOLD", "PRUNE")

    def __init__(
        self,
        perturbation_fraction: float = 0.50,
        prune_fraction: float = 0.10,
        dwell_probes: int = 2,
        gradient_window: int = 4,
        min_warmup_probes: int = 3,
        recheck_probes: int = 10,
        noise_deadband: float = 0.05,
        probe_every_k_refines: int = 5,
    ):
        if perturbation_fraction <= 0.0 or perturbation_fraction >= 1.0:
            raise ValueError(f"perturbation_fraction must be in (0, 1), got {perturbation_fraction}")
        if prune_fraction <= 0.0 or prune_fraction >= 1.0:
            raise ValueError(f"prune_fraction must be in (0, 1), got {prune_fraction}")
        if dwell_probes < 1:
            raise ValueError(f"dwell_probes must be >= 1, got {dwell_probes}")
        if gradient_window < 2:
            raise ValueError(f"gradient_window must be >= 2, got {gradient_window}")
        if gradient_window <= dwell_probes:
            raise ValueError(
                f"gradient_window ({gradient_window}) must be > dwell_probes ({dwell_probes}) "
                f"so the estimation window spans across the perturbation burst"
            )
        if not (0.0 <= noise_deadband < 1.0):
            raise ValueError(f"noise_deadband must be in [0, 1), got {noise_deadband}")
        if probe_every_k_refines < 1:
            raise ValueError(f"probe_every_k_refines must be >= 1, got {probe_every_k_refines}")

        self._perturbation_fraction = perturbation_fraction
        self._prune_fraction = prune_fraction
        self._dwell_probes = dwell_probes
        self._gradient_window = gradient_window
        self._min_warmup_probes = min_warmup_probes
        self._recheck_probes = recheck_probes
        self._noise_deadband = noise_deadband
        self._probe_every_k = probe_every_k_refines

        # Per-step rates so the TOTAL burst over probe_every_k_refines steps
        # equals the desired fraction (avoids exponential compounding).
        self._insert_rate = (1.0 + perturbation_fraction) ** (1.0 / probe_every_k_refines)
        self._prune_rate = (1.0 - prune_fraction) ** (1.0 / probe_every_k_refines)

        self._state: str = "WARMUP"
        self._history: list[tuple[int, float, int]] = []
        self._current_rate: float = 1.0

        self._dwell_counter: int = 0
        self._hold_counter: int = 0
        self._peak_abs_gradient: float = 0.0
        self._last_gaussian_count: int = 0
        self._last_gradient: float | None = None

        # Baseline slope (dMetric/dStep from HOLD phase, for HP-filter subtraction)
        self._baseline_slope: float = 0.0

        # Direction of the last perturbation: +1 for INSERT, -1 for PRUNE
        self._last_perturbation_sign: int = 1

        # Pre-burst reference for paired-difference gradient estimation:
        # (step, metric, gc) recorded at the INSERT/PRUNE observation.
        self._burst_ref: tuple[int, float, int] | None = None

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def history(self) -> list[tuple[int, float, int]]:
        return list(self._history)

    @property
    def perturbation_size(self) -> int:
        """Number of Gaussians to add or remove in the current burst."""
        return max(1, int(self._last_gaussian_count * self._perturbation_fraction))

    @property
    def baseline_slope(self) -> float:
        return self._baseline_slope

    @property
    def last_gradient(self) -> float | None:
        return self._last_gradient

    @property
    def is_saturated(self) -> bool:
        """ESC has no saturation concept; always False for interface compatibility."""
        return False

    @property
    def is_cooling_down(self) -> bool:
        """Returns True when in DWELL (analogous to cooldown in MetricController)."""
        return self._state == "DWELL"

    @property
    def peak_effectiveness(self) -> float:
        """Returns peak absolute gradient for logging compatibility."""
        return self._peak_abs_gradient

    def get_insertion_rate(self) -> float:
        return self._current_rate

    # ------------------------------------------------------------------
    # Core observe method
    # ------------------------------------------------------------------

    def observe(self, step: int, metric_value: float, gaussian_count: int = 0) -> float:
        """
        Record a metric observation and advance the state machine.

        Returns:
            The insertion rate: > 1.0 for INSERT, 1.0 for HOLD/DWELL, < 1.0 for PRUNE.
        """
        self._history.append((step, metric_value, gaussian_count))
        if gaussian_count > 0:
            self._last_gaussian_count = gaussian_count

        if self._state == "WARMUP":
            self._handle_warmup()
        elif self._state == "INSERT":
            self._handle_insert()
        elif self._state == "PRUNE":
            self._handle_prune()
        elif self._state == "DWELL":
            self._handle_dwell()
        elif self._state == "ESTIMATE":
            self._handle_estimate()
        elif self._state == "HOLD":
            self._handle_hold()

        self._logger.info(
            f"ESC: step={step}, metric={metric_value:.4f}, gc={gaussian_count:,}, "
            f"state={self._state}, rate={self._current_rate:.4f}, "
            f"gradient={self._last_gradient}, baseline={self._baseline_slope:.2e}"
        )
        return self._current_rate

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_warmup(self) -> None:
        self._current_rate = 1.0
        if len(self._history) >= self._min_warmup_probes:
            self._capture_baseline()
            self._transition_to("INSERT")

    def _handle_insert(self) -> None:
        self._burst_ref = self._history[-1]
        self._current_rate = self._insert_rate
        self._last_perturbation_sign = 1
        self._enter_dwell()

    def _handle_prune(self) -> None:
        self._burst_ref = self._history[-1]
        self._current_rate = self._prune_rate
        self._last_perturbation_sign = -1
        self._enter_dwell()

    def _enter_dwell(self) -> None:
        self._dwell_counter = 0
        self._transition_to("DWELL")

    def _handle_dwell(self) -> None:
        self._current_rate = 1.0
        self._dwell_counter += 1
        if self._dwell_counter >= self._dwell_probes:
            self._capture_baseline_from_dwell()
            self._transition_to("ESTIMATE")

    def _capture_baseline_from_dwell(self) -> None:
        """Update HP-filter baseline from DWELL observations (constant gc = pure optimization rate)."""
        window = self._history[-self._dwell_probes :]
        if len(window) >= 2:
            new_baseline = MetricController._linear_regression_slope([(s, m) for s, m, _gc in window])
            self._logger.info(
                f"ESC baseline update: {self._baseline_slope:.2e} -> {new_baseline:.2e} "
                f"(from {len(window)} DWELL observations)"
            )
            self._baseline_slope = new_baseline

    def _handle_estimate(self) -> None:
        gradient = self._estimate_gradient()
        self._last_gradient = gradient

        if gradient is None:
            self._logger.info(
                "ESC ESTIMATE: gradient indeterminate (insufficient gc variation), " "scheduling exploratory INSERT"
            )
            self._current_rate = 1.0
            self._transition_to("INSERT")
            return

        abs_grad = abs(gradient)
        if abs_grad > self._peak_abs_gradient:
            self._peak_abs_gradient = abs_grad

        deadband_threshold = self._peak_abs_gradient * self._noise_deadband

        if self._peak_abs_gradient > 0.0 and abs_grad < deadband_threshold:
            self._capture_baseline()
            self._hold_counter = 0
            self._current_rate = 1.0
            self._logger.info(
                f"ESC ESTIMATE: gradient={gradient:.2e} within deadband "
                f"(threshold={deadband_threshold:.2e}), entering HOLD"
            )
            self._transition_to("HOLD")
        elif gradient > 0:
            self._current_rate = 1.0
            self._logger.info(f"ESC ESTIMATE: gradient={gradient:.2e} > 0, scheduling INSERT")
            self._transition_to("INSERT")
        else:
            self._current_rate = 1.0
            self._logger.info(f"ESC ESTIMATE: gradient={gradient:.2e} < 0, scheduling PRUNE")
            self._transition_to("PRUNE")

    def _handle_hold(self) -> None:
        self._current_rate = 1.0
        self._hold_counter += 1
        if self._hold_counter >= self._recheck_probes:
            self._transition_to("ESTIMATE")

    def _transition_to(self, new_state: str) -> None:
        if new_state not in self.STATES:
            raise ValueError(f"Invalid state: {new_state}")
        old = self._state
        self._state = new_state
        if old != new_state:
            self._logger.info(f"ESC transition: {old} -> {new_state}")

    # ------------------------------------------------------------------
    # Gradient estimation
    # ------------------------------------------------------------------

    def _estimate_gradient(self) -> float | None:
        """
        Estimate dQuality/dGaussianCount using a paired-difference approach.

        Compares the pre-burst reference observation to the average DWELL
        observation, with baseline (optimization-only) subtraction applied
        to the DWELL period.  This avoids mixing observations from different
        optimization phases that would confound a regression.

        Falls back to regression over the gradient window when no burst
        reference is available (e.g. after HOLD -> ESTIMATE).

        Returns None when the Gaussian count hasn't varied enough.
        """
        if self._burst_ref is not None:
            return self._estimate_gradient_paired()
        return self._estimate_gradient_regression()

    def _estimate_gradient_paired(self) -> float | None:
        """Paired-difference gradient: compare pre-burst ref to DWELL average.

        No baseline correction is applied here because with short DWELL periods
        (2-3 probes) the baseline estimate is noisier than the signal it tries
        to correct.  The optimization-only improvement over such a short span
        is small relative to the densification signal.
        """
        ref_step, ref_metric, ref_gc = self._burst_ref
        dwell_obs = self._history[-self._dwell_probes :]
        if len(dwell_obs) < 1:
            return None

        dwell_gc = dwell_obs[0][2]
        delta_gc = dwell_gc - ref_gc
        if delta_gc == 0:
            return None

        avg_step = sum(s for s, _m, _gc in dwell_obs) / len(dwell_obs)
        avg_metric = sum(m for _s, m, _gc in dwell_obs) / len(dwell_obs)

        gradient = (avg_metric - ref_metric) / delta_gc
        self._logger.info(
            f"ESC gradient (paired): ref=({ref_step}, {ref_metric:.4f}, {ref_gc:,}), "
            f"dwell_avg=({avg_step:.0f}, {avg_metric:.4f}, {dwell_gc:,}), "
            f"delta_gc={delta_gc:,}"
        )
        return gradient

    def _estimate_gradient_regression(self) -> float | None:
        """Regression fallback for HOLD -> ESTIMATE (no burst reference)."""
        window = self._history[-self._gradient_window :]
        if len(window) < 2:
            return None

        gc_values = [gc for _s, _m, gc in window]
        gc_min, gc_max = min(gc_values), max(gc_values)
        if gc_max <= gc_min:
            return None

        step_0 = window[0][0]
        adjusted = [(float(gc), metric - self._baseline_slope * (step - step_0)) for step, metric, gc in window]

        return MetricController._linear_regression_slope(adjusted)

    def _capture_baseline(self) -> None:
        """Capture the HOLD-phase dMetric/dStep as baseline for HP-filter subtraction."""
        window = self._history[-self._gradient_window :]
        if len(window) >= 2:
            self._baseline_slope = MetricController._linear_regression_slope([(s, m) for s, m, _gc in window])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "perturbation_fraction": self._perturbation_fraction,
            "prune_fraction": self._prune_fraction,
            "dwell_probes": self._dwell_probes,
            "gradient_window": self._gradient_window,
            "min_warmup_probes": self._min_warmup_probes,
            "recheck_probes": self._recheck_probes,
            "noise_deadband": self._noise_deadband,
            "probe_every_k": self._probe_every_k,
            "state": self._state,
            "history": self._history,
            "current_rate": self._current_rate,
            "dwell_counter": self._dwell_counter,
            "hold_counter": self._hold_counter,
            "peak_abs_gradient": self._peak_abs_gradient,
            "last_gaussian_count": self._last_gaussian_count,
            "baseline_slope": self._baseline_slope,
            "last_perturbation_sign": self._last_perturbation_sign,
            "last_gradient": self._last_gradient,
            "burst_ref": self._burst_ref,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._perturbation_fraction = state_dict["perturbation_fraction"]
        self._prune_fraction = state_dict.get("prune_fraction", self._perturbation_fraction)
        self._dwell_probes = state_dict["dwell_probes"]
        self._gradient_window = state_dict["gradient_window"]
        self._min_warmup_probes = state_dict["min_warmup_probes"]
        self._recheck_probes = state_dict["recheck_probes"]
        self._noise_deadband = state_dict["noise_deadband"]
        self._probe_every_k = state_dict.get("probe_every_k", 5)
        self._insert_rate = (1.0 + self._perturbation_fraction) ** (1.0 / self._probe_every_k)
        self._prune_rate = (1.0 - self._prune_fraction) ** (1.0 / self._probe_every_k)
        self._state = state_dict["state"]
        self._history = state_dict["history"]
        self._current_rate = state_dict["current_rate"]
        self._dwell_counter = state_dict.get("dwell_counter", 0)
        self._hold_counter = state_dict.get("hold_counter", 0)
        self._peak_abs_gradient = state_dict.get("peak_abs_gradient", 0.0)
        self._last_gaussian_count = state_dict.get("last_gaussian_count", 0)
        self._baseline_slope = state_dict.get("baseline_slope", 0.0)
        self._last_perturbation_sign = state_dict.get("last_perturbation_sign", 1)
        self._last_gradient = state_dict.get("last_gradient", None)
        raw_ref = state_dict.get("burst_ref")
        self._burst_ref = tuple(raw_ref) if raw_ref is not None else None


@dataclass
class GaussianSplatOptimizerMCMCControlledConfig(GaussianSplatOptimizerConfig):
    """
    Configuration for ``GaussianSplatOptimizerMCMCControlled``.

    Extends the standard MCMC optimizer with metric-based feedback control
    of the insertion rate.
    """

    # ---- Standard MCMC parameters ----

    # Override base class defaults to match the MCMC paper:
    # "3D Gaussian Splatting as Markov Chain Monte Carlo" (https://arxiv.org/abs/2404.09591)
    initial_opacity: float = 0.5
    """
    Initial opacity of each Gaussian for MCMC optimization.

    Default: ``0.5`` (matches the MCMC paper, different from base 3DGS default of 0.1).
    """

    initial_covariance_scale: float = 0.1
    """
    Initial scale of each Gaussian for MCMC optimization.

    Default: ``0.1`` (matches the MCMC paper, different from base 3DGS default of 1.0).
    """

    noise_lr: float = 5e5
    """
    Learning rate for the noise added to Gaussian positions.

    Default: ``5e5``.
    """

    insertion_rate: float = 1.05
    """
    Maximum insertion rate used by the controller when quality has stalled.
    In controlled mode, this serves as the upper bound for the controller output.

    Default: ``1.05`` (5% growth per refinement step).
    """

    binomial_coeffs_n_max: int = 51
    """
    Maximum replication ratio for the MCMC relocation kernel.

    Default: ``51``.
    """

    opacity_regularization: float = 0.01
    """
    Weight for opacity regularization loss.

    Default: ``0.01``.
    """

    scale_regularization: float = 0.01
    """
    Weight for scale regularization loss.

    Default: ``0.01``.
    """

    # ---- Metric controller parameters ----

    control_metric: str = "ssim"
    """
    Which quality metric drives the controller: ``"psnr"`` or ``"ssim"``.

    The probe renders validation images and computes both metrics; this
    setting selects which one is fed into the trend estimator.  SSIM
    provides a smoother, less noisy signal than PSNR, making stall
    detection more reliable.

    Default: ``"ssim"``.
    """

    probe_every_k_refines: int = 5
    """
    Measure the quality metric every K refinement steps.

    Default: ``5``.
    """

    probe_n_images: int = 3
    """
    Number of random validation images to render per probe.

    Default: ``3``.
    """

    trend_window: int = 5
    """
    Number of recent metric measurements used for the linear trend fit.

    Default: ``5``.
    """

    slope_threshold: float = 1e-6
    """
    Slope of the metric trend (dMetric/dStep) below which improvement is
    considered "stalled" and the controller triggers insertion (HOLD -> INSERT).

    The appropriate scale depends on ``control_metric``: ~1e-4 for PSNR
    (dB/step), ~1e-6 for SSIM (unitless/step).

    Default: ``1e-6`` (tuned for SSIM).
    """

    slope_threshold_high: float = 2e-6
    """
    Slope of the metric trend (dMetric/dStep) above which the controller
    stops inserting (INSERT -> HOLD).  Must be >= ``slope_threshold``.
    The gap between the two thresholds creates a hysteresis dead band
    that prevents chattering.  Set equal to ``slope_threshold`` to
    disable hysteresis.

    Default: ``2e-6`` (tuned for SSIM).
    """

    effectiveness_threshold: float = 0.0
    """
    Minimum dMetric / dGaussians for an insertion burst to be considered
    effective (anti-windup).  When a burst ends and the measured
    effectiveness falls below this threshold, the controller permanently
    suppresses further insertion.

    Set to ``0.0`` to disable anti-windup (default for backward
    compatibility).

    Default: ``0.0`` (disabled).
    """

    min_measurements_before_control: int = 3
    """
    Number of metric observations to collect before the controller activates.
    During warmup, the controller uses the maximum insertion rate.

    Default: ``3``.
    """

    cooldown_probes: int = 0
    """
    Number of probe intervals to force HOLD after each INSERT->HOLD transition.

    After a burst of insertion, newly added Gaussians need gradient steps to
    be optimised.  Continuing to insert immediately can disrupt optimisation
    and create a positive feedback loop (insertion-induced stall triggers more
    insertion).  The cooldown breaks this loop by enforcing a recovery period.

    Set to ``0`` to disable cooldown (default for backward compatibility).

    Default: ``0`` (disabled).
    """

    max_new_gaussians_per_refine: int = 0
    """
    Maximum number of new Gaussians that can be added in a single refinement
    step, regardless of the controller's insertion rate.

    Because the insertion rate is multiplicative (e.g. 5% of current count),
    the absolute number of new Gaussians per step grows with the model size.
    At 300K Gaussians, 5% adds 15K (gentle); at 5M, 5% adds 250K (disruptive).
    This cap converts exponential growth into linear growth once the model
    exceeds ``max_new_gaussians_per_refine / (insertion_rate - 1)`` Gaussians.

    Set to ``0`` to disable the cap (default for backward compatibility).

    Default: ``0`` (disabled).
    """

    effectiveness_decay_ratio: float = 0.0
    """
    Fraction of peak insertion effectiveness (dMetric/dGaussians) below which
    the controller transitions from INSERT to HOLD.  Enables automatic early
    growth with diminishing-returns plateau detection.

    When enabled (> 0), the controller defaults to INSERT after warmup and
    continuously monitors how much each new Gaussian improves quality.  Once
    effectiveness drops to this fraction of the peak observed value, the
    controller switches to HOLD.  The existing slope-based stall detection
    still handles HOLD->INSERT recovery.

    Set to ``0.0`` to disable and use the original slope-based INSERT->HOLD
    logic (default for backward compatibility).

    Default: ``0.0`` (disabled).
    """

    # ---- Controller type selection ----

    controller_type: str = "metric"
    """
    Which controller to use: ``"metric"`` for the existing MetricController,
    ``"esc"`` for the ExtremumSeekingController.

    Default: ``"metric"`` (backward compatible).
    """

    # ---- ESC-specific parameters (only used when controller_type == "esc") ----

    perturbation_fraction: float = 0.50
    """
    Fraction of current Gaussian count to ADD per INSERT burst.
    The per-step rate is computed as ``(1 + f)^(1/probe_every_k_refines)`` to avoid
    exponential compounding.

    Default: ``0.50``.
    """

    prune_fraction: float = 0.10
    """
    Fraction of current Gaussian count to REMOVE per PRUNE burst.
    Kept smaller than perturbation_fraction for asymmetric control: grow fast,
    correct gently.

    Default: ``0.10``.
    """

    dwell_probes: int = 2
    """
    Number of probe intervals to wait after a perturbation before estimating
    the gradient.  Enforces time-scale separation.

    Default: ``2``.
    """

    gradient_window: int = 4
    """
    Number of observations for gradient estimation (regression fallback).
    Must be > dwell_probes so the window spans across the perturbation burst.

    Default: ``4``.
    """

    min_warmup_probes: int = 3
    """
    Warmup observations before ESC control activates.  INSERT-by-default
    during warmup.

    Default: ``3``.
    """

    recheck_probes: int = 10
    """
    Probes between gradient re-estimates during HOLD.

    Default: ``10``.
    """

    noise_deadband: float = 0.05
    """
    Gradient magnitudes below this fraction of peak observed |gradient| are
    treated as near-zero (HOLD).  Self-normalizing against the scene's own
    gradient scale.

    Default: ``0.05``.
    """

    def make_optimizer(self, model: GaussianSplat3d, sfm_scene: SfmScene) -> "GaussianSplatOptimizerMCMCControlled":
        return GaussianSplatOptimizerMCMCControlled.from_model_and_scene(
            model=model,
            sfm_scene=sfm_scene,
            config=self,
        )


@splat_optimizer
class GaussianSplatOptimizerMCMCControlled(BaseGaussianSplatOptimizer):
    """
    MCMC optimizer with metric-controlled insertion rate.

    This optimizer extends the standard MCMC approach by replacing the fixed insertion
    rate with a feedback controller.  A lightweight quality probe (PSNR or SSIM,
    controlled by ``control_metric``) is evaluated every K refinement steps and the
    resulting trend drives insertion decisions:

    * When quality is improving (positive slope) -> hold steady (``insertion_rate = 1.0``)
    * When quality has stalled (slope < threshold) -> insert (``insertion_rate = max_rate``)

    The probe and control loop are driven externally via :meth:`observe_metric` which
    should be called from the reconstruction training loop after each probe evaluation.

    .. note:: Create instances via :meth:`from_model_and_scene` or :meth:`from_state_dict`.
    """

    __PRIVATE__ = object()

    def __init__(
        self,
        model: GaussianSplat3d,
        config: GaussianSplatOptimizerMCMCControlledConfig,
        optimizer: torch.optim.Adam,
        spatial_scale: float,
        refine_count: int,
        step_count: int,
        controller: MetricController | None = None,
        _private: Any = None,
    ):
        if _private is not self.__PRIVATE__:
            raise RuntimeError(
                "GaussianSplatOptimizerMCMCControlled must be created using "
                "from_model_and_scene() or from_state_dict()"
            )
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._step_count = step_count
        self._refine_count = refine_count
        self._spatial_scale = spatial_scale
        self._config = config
        self._model = model
        self._optimizer = optimizer

        # Learning rate decay for means
        self._means_lr_decay_exponent = 1.0
        self._means_lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._binomial_coeffs: torch.Tensor | None = None
        self._binomial_coeffs_n_max: int | None = None

        # Controller (MetricController or ExtremumSeekingController)
        self._controller: MetricController | ExtremumSeekingController = controller or self._make_controller(config)

        self._ensure_means_lr_scheduler()

    @staticmethod
    def _make_controller(
        config: "GaussianSplatOptimizerMCMCControlledConfig",
    ) -> "MetricController | ExtremumSeekingController":
        if config.controller_type == "esc":
            return ExtremumSeekingController(
                perturbation_fraction=config.perturbation_fraction,
                prune_fraction=config.prune_fraction,
                dwell_probes=config.dwell_probes,
                gradient_window=config.gradient_window,
                min_warmup_probes=config.min_warmup_probes,
                recheck_probes=config.recheck_probes,
                noise_deadband=config.noise_deadband,
                probe_every_k_refines=config.probe_every_k_refines,
            )
        return MetricController(
            trend_window=config.trend_window,
            slope_threshold=config.slope_threshold,
            slope_threshold_high=config.slope_threshold_high,
            max_insertion_rate=config.insertion_rate,
            min_measurements_before_control=config.min_measurements_before_control,
            effectiveness_threshold=config.effectiveness_threshold,
            cooldown_probes=config.cooldown_probes,
            effectiveness_decay_ratio=config.effectiveness_decay_ratio,
        )

    # ------------------------------------------------------------------
    # Public API: metric feedback
    # ------------------------------------------------------------------

    @property
    def control_metric(self) -> str:
        """Which quality metric drives the controller (``"psnr"`` or ``"ssim"``)."""
        return self._config.control_metric

    @property
    def probe_every_k_refines(self) -> int:
        """How often (in refinement steps) the reconstruction loop should probe metrics."""
        return self._config.probe_every_k_refines

    @property
    def probe_n_images(self) -> int:
        """Number of validation images to render per probe."""
        return self._config.probe_n_images

    @property
    def controller(self) -> "MetricController | ExtremumSeekingController":
        """Access the underlying controller (read-only)."""
        return self._controller

    def observe_metric(self, step: int, metric_value: float) -> float:
        """
        Push a metric observation from the reconstruction loop into the controller.

        Args:
            step: The global training step.
            metric_value: The measured quality metric (PSNR or SSIM, depending
                on ``control_metric``).

        Returns:
            The updated insertion rate that will be used on the next refinement step.
        """
        return self._controller.observe(step, metric_value, gaussian_count=self._model.num_gaussians)

    # ------------------------------------------------------------------
    # Optimizer lifecycle (mirrors GaussianSplatOptimizerMCMC)
    # ------------------------------------------------------------------

    def _ensure_means_lr_scheduler(self) -> None:
        if self._means_lr_scheduler is not None:
            return

        def _one(_: int) -> float:
            return 1.0

        def _means_lambda(_: int, _self: "GaussianSplatOptimizerMCMCControlled" = self) -> float:
            return float(_self._means_lr_decay_exponent)

        lr_lambdas: list[Callable[[int], float]] = []
        for pg in self._optimizer.param_groups:
            if pg.get("name") == "means":
                lr_lambdas.append(_means_lambda)
            else:
                lr_lambdas.append(_one)

        self._means_lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self._optimizer,
            lr_lambda=lr_lambdas,
        )

    def _get_binomial_coeffs(self) -> tuple[torch.Tensor, int]:
        n_max = int(self._config.binomial_coeffs_n_max)
        if n_max <= 0:
            raise ValueError("n_max must be > 0")
        if (
            self._binomial_coeffs is None
            or self._binomial_coeffs_n_max != n_max
            or self._binomial_coeffs.device != self._model.device
        ):
            self._binomial_coeffs = self._build_binomial_coeffs(n_max=n_max, device=self._model.device)
            self._binomial_coeffs_n_max = n_max
        return self._binomial_coeffs, n_max

    @staticmethod
    def _build_binomial_coeffs(n_max: int, device: torch.device) -> torch.Tensor:
        coeffs = torch.zeros((n_max, n_max), device=device, dtype=torch.float32)
        for row in range(n_max):
            coeffs[row, 0] = 1.0
            coeffs[row, row] = 1.0
            for k in range(1, row):
                coeffs[row, k] = coeffs[row - 1, k - 1] + coeffs[row - 1, k]
        return coeffs

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_model_and_scene(
        cls,
        model: GaussianSplat3d,
        sfm_scene: SfmScene,
        config: GaussianSplatOptimizerMCMCControlledConfig = GaussianSplatOptimizerMCMCControlledConfig(),
    ) -> "GaussianSplatOptimizerMCMCControlled":
        spatial_scale = sfm_scene.spatial_scale(config.spatial_scale_mode) * config.spatial_scale_multiplier
        optimizer = GaussianSplatOptimizer._make_optimizer(model, spatial_scale, config)

        return cls(
            model=model,
            optimizer=optimizer,
            config=config,
            spatial_scale=spatial_scale,
            refine_count=0,
            step_count=0,
            _private=cls.__PRIVATE__,
        )

    @classmethod
    def from_state_dict(
        cls, model: GaussianSplat3d, state_dict: dict[str, Any]
    ) -> "GaussianSplatOptimizerMCMCControlled":
        if "version" not in state_dict:
            raise ValueError("State dict is missing version information")
        if state_dict["version"] not in (1,):
            raise ValueError(f"Unsupported version: {state_dict['version']}")

        config = GaussianSplatOptimizerMCMCControlledConfig(**state_dict["config"])

        adam_optimizer = GaussianSplatOptimizer._make_optimizer(model=model, means_lr_scale=1.0, config=config)
        adam_optimizer.load_state_dict(state_dict["optimizer"])

        # Restore the controller (type determined by config)
        controller = cls._make_controller(config)
        if "controller" in state_dict:
            controller.load_state_dict(state_dict["controller"])

        opt = cls(
            model=model,
            optimizer=adam_optimizer,
            spatial_scale=state_dict["spatial_scale"],
            config=config,
            step_count=state_dict["step_count"],
            refine_count=state_dict["refine_count"],
            controller=controller,
            _private=cls.__PRIVATE__,
        )
        opt._means_lr_decay_exponent = state_dict["means_lr_decay_exponent"]
        opt._ensure_means_lr_scheduler()
        return opt

    # ------------------------------------------------------------------
    # Core optimizer methods
    # ------------------------------------------------------------------

    def step(self):
        """Step the optimizer, adding noise to Gaussian positions (MCMC exploration)."""
        means_lr: float | None = None
        for pg in self._optimizer.param_groups:
            if pg.get("name") == "means":
                means_lr = float(pg["lr"])
                break
        if means_lr is None:
            raise RuntimeError("Could not find 'means' param group in optimizer")
        noise_scale = float(self._config.noise_lr) * means_lr
        if noise_scale != 0.0:
            self._model.add_noise_to_means(noise_scale=noise_scale)

        self._optimizer.step()
        self._step_count += 1
        self._ensure_means_lr_scheduler()
        assert self._means_lr_scheduler is not None
        self._means_lr_scheduler.step()

    def zero_grad(self, set_to_none: bool = False):
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def _estimate_bytes_per_gaussian(self) -> int:
        """
        Estimate GPU memory required per Gaussian (parameters + Adam optimizer state).

        Each parameter group contributes: element_size * elements_per_gaussian * 3
        (the factor of 3 accounts for the parameter itself plus Adam's exp_avg and exp_avg_sq).
        """
        total = 0
        for pg in self._optimizer.param_groups:
            param = pg["params"][0]
            elems_per_gaussian = param[0].numel()
            total += 3 * param.element_size() * elems_per_gaussian
        return total

    def _cap_insertion_to_available_memory(self, num_added: int, safety_factor: float = 3.0) -> int:
        """
        Reduce ``num_added`` if the GPU does not have enough free memory.

        Uses ``torch.cuda.mem_get_info`` to query free memory, estimates the cost
        of ``num_added`` new Gaussians (params + optimizer state), and applies a
        ``safety_factor`` multiplier to account for temporary allocations during
        ``_sample_add`` (multinomial sampling, binomial coefficients, concat buffers).

        Args:
            num_added: The desired number of Gaussians to add.
            safety_factor: Multiplier on the per-Gaussian cost estimate.

        Returns:
            The (possibly reduced) number of Gaussians that can safely be added.
        """
        if num_added <= 0 or self._model.device.type != "cuda":
            return num_added

        free_mem, _ = torch.cuda.mem_get_info(self._model.device)
        bytes_per_gaussian = self._estimate_bytes_per_gaussian()
        cost_per_gaussian = int(bytes_per_gaussian * safety_factor)

        if cost_per_gaussian <= 0:
            return num_added

        max_addable = free_mem // cost_per_gaussian

        if max_addable <= 0:
            self._logger.warning(
                f"GPU memory too low to add any Gaussians "
                f"(free={free_mem / 1e9:.2f} GB, cost/gaussian={cost_per_gaussian} B). "
                f"Skipping insertion of {num_added:,} Gaussians."
            )
            return 0

        if max_addable < num_added:
            self._logger.info(
                f"Capping insertion from {num_added:,} to {max_addable:,} Gaussians "
                f"(free={free_mem / 1e9:.2f} GB, est. cost={num_added * cost_per_gaussian / 1e9:.2f} GB)."
            )
            return int(max_addable)

        return num_added

    @torch.no_grad()
    def refine(self, zero_gradients: bool = True) -> dict[str, int | float]:
        """
        Perform a refinement step using the controller-determined insertion rate.

        When the rate is > 1.0, new Gaussians are added (INSERT).
        When the rate is < 1.0, low-opacity Gaussians are pruned (PRUNE).
        When the rate is == 1.0, the model is unchanged (HOLD/DWELL).
        """
        num_gaussians_before = self._model.num_gaussians

        # Relocate dead Gaussians
        num_relocated = self._relocate()

        # Use the controller's insertion rate
        insertion_rate = self._controller.get_insertion_rate()

        num_added = 0
        num_pruned = 0

        if insertion_rate > 1.0:
            # INSERT: grow the Gaussian population
            if self._config.max_gaussians > 0:
                num_target = min(self._config.max_gaussians, int(insertion_rate * self._model.num_gaussians))
            else:
                num_target = int(insertion_rate * self._model.num_gaussians)
            num_added = max(0, num_target - self._model.num_gaussians)

            if self._config.max_new_gaussians_per_refine > 0 and num_added > self._config.max_new_gaussians_per_refine:
                num_added = self._config.max_new_gaussians_per_refine

            num_added = self._cap_insertion_to_available_memory(num_added)

            if num_added > 0:
                self._sample_add(num_added)

        elif insertion_rate < 1.0:
            # PRUNE: shrink the Gaussian population
            num_to_prune = self._model.num_gaussians - int(insertion_rate * self._model.num_gaussians)
            if num_to_prune > 0:
                num_pruned = self._prune(num_to_prune)

        if zero_gradients:
            self._model.log_scales.grad = None
            self._model.logit_opacities.grad = None
            self._model.quats.grad = None
            self._model.means.grad = None
            self._model.sh0.grad = None
            self._model.shN.grad = None

        self._refine_count += 1
        self._logger.debug(
            f"Controlled MCMC refinement (step {self._step_count:,}): "
            f"{num_relocated:,} relocated, {num_added:,} added, {num_pruned:,} pruned "
            f"(rate={insertion_rate:.4f}). "
            f"Before: {num_gaussians_before:,}, After: {self._model.num_gaussians:,}"
        )

        return {
            "num_relocated": num_relocated,
            "num_added": num_added,
            "num_pruned": num_pruned,
            "insertion_rate": insertion_rate,
        }

    def regularization_loss(self) -> torch.Tensor:
        loss = self._config.opacity_regularization * self._model.opacities.mean()
        loss = loss + self._config.scale_regularization * self._model.scales.mean()
        return loss

    # ------------------------------------------------------------------
    # Gaussian manipulation (same as GaussianSplatOptimizerMCMC)
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _multinomial_sample(weights: torch.Tensor, n: int, replacement: bool = True) -> torch.Tensor:
        num_elements = weights.size(0)
        if num_elements <= 2**24:
            return torch.multinomial(weights, n, replacement=replacement)
        else:
            weights = weights / weights.sum()
            weights_np = weights.detach().cpu().numpy()
            sampled_idxs_np = np.random.choice(num_elements, size=n, p=weights_np, replace=replacement)
            sampled_idxs = torch.from_numpy(sampled_idxs_np)
            return sampled_idxs.to(weights.device)

    @torch.no_grad()
    def _update_optimizer_params_and_state(
        self,
        optimizer_fn: Callable[[torch.Tensor], torch.Tensor],
        parameter_names: set[str] | None = None,
        reset_adam_step_counts: bool = False,
    ):
        for i, param_group in enumerate(self._optimizer.param_groups):
            parameter_name = param_group["name"]
            if parameter_names is not None and parameter_name not in parameter_names:
                continue
            assert len(param_group["params"]) == 1, "Expected one parameter tensor per param group"
            old_parameter = param_group["params"][0]
            optimizer_state = self._optimizer.state[old_parameter]
            del self._optimizer.state[old_parameter]
            for key, value in optimizer_state.items():
                if key != "step":
                    optimizer_state[key] = optimizer_fn(value)
                elif reset_adam_step_counts:
                    optimizer_state[key].zero_()
            new_parameter = getattr(self._model, parameter_name)
            new_parameter.requires_grad = True
            self._optimizer.state[new_parameter] = optimizer_state
            self._optimizer.param_groups[i]["params"] = [new_parameter]

        if self._model.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _relocate(self) -> int:
        dead_mask = self._model.opacities <= self._config.deletion_opacity_threshold
        n_gs = int(dead_mask.sum().item())
        if n_gs > 0:
            dead_indices = dead_mask.nonzero(as_tuple=True)[0]
            alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
            n = len(dead_indices)

            probs = self._model.opacities[alive_indices].flatten()
            if probs.numel() == 0:
                return 0
            if float(probs.sum().item()) == 0.0:
                probs = torch.ones_like(probs)
            sampled_idxs = self._multinomial_sample(probs, n, replacement=True)
            sampled_idxs = alive_indices[sampled_idxs]
            ratios = torch.bincount(sampled_idxs, minlength=self._model.num_gaussians)[sampled_idxs] + 1
            binomial_coeffs, n_max = self._get_binomial_coeffs()
            ratios = ratios.to(dtype=torch.int32)
            new_logit_opacities, new_log_scales = self._model.relocate_gaussians(
                log_scales=self._model.log_scales[sampled_idxs],
                logit_opacities=self._model.logit_opacities[sampled_idxs],
                ratios=ratios,
                binomial_coeffs=binomial_coeffs,
                n_max=n_max,
                min_opacity=self._config.deletion_opacity_threshold,
            )

            self._model.log_scales[sampled_idxs] = new_log_scales
            self._model.logit_opacities[sampled_idxs] = new_logit_opacities
            for param_name in ["log_scales", "logit_opacities", "quats", "means", "sh0", "shN"]:
                param = getattr(self._model, param_name)
                param[dead_indices] = param[sampled_idxs]

            def zero_sampled_gradients(x: torch.Tensor) -> torch.Tensor:
                x[sampled_idxs] = 0
                return x

            self._update_optimizer_params_and_state(
                optimizer_fn=zero_sampled_gradients,
                parameter_names={"log_scales", "logit_opacities", "quats", "means", "sh0", "shN"},
                reset_adam_step_counts=False,
            )
        return n_gs

    @torch.no_grad()
    def _sample_add(self, n: int) -> int:
        probs = self._model.opacities.flatten()
        if probs.numel() == 0:
            return 0
        if float(probs.sum().item()) == 0.0:
            probs = torch.ones_like(probs)
        sampled_idxs = self._multinomial_sample(probs, n, replacement=True)
        ratios = torch.bincount(sampled_idxs, minlength=self._model.num_gaussians)[sampled_idxs] + 1
        binomial_coeffs, n_max = self._get_binomial_coeffs()
        ratios = ratios.to(dtype=torch.int32)
        new_logit_opacities, new_log_scales = self._model.relocate_gaussians(
            log_scales=self._model.log_scales[sampled_idxs],
            logit_opacities=self._model.logit_opacities[sampled_idxs],
            ratios=ratios,
            binomial_coeffs=binomial_coeffs,
            n_max=n_max,
            min_opacity=self._config.deletion_opacity_threshold,
        )

        self._model.log_scales[sampled_idxs] = new_log_scales
        self._model.logit_opacities[sampled_idxs] = new_logit_opacities

        self._model.set_state(
            means=torch.cat([self._model.means, self._model.means[sampled_idxs]], dim=0),
            quats=torch.cat([self._model.quats, self._model.quats[sampled_idxs]], dim=0),
            log_scales=torch.cat([self._model.log_scales, self._model.log_scales[sampled_idxs]], dim=0),
            logit_opacities=torch.cat([self._model.logit_opacities, self._model.logit_opacities[sampled_idxs]], dim=0),
            sh0=torch.cat([self._model.sh0, self._model.sh0[sampled_idxs]], dim=0),
            shN=torch.cat([self._model.shN, self._model.shN[sampled_idxs]], dim=0),
        )

        def zero_extend_sampled_gradients(x: torch.Tensor) -> torch.Tensor:
            x = torch.cat([x, torch.zeros(n, *x.shape[1:], dtype=x.dtype, device=x.device)])
            return x

        self._update_optimizer_params_and_state(
            optimizer_fn=zero_extend_sampled_gradients,
            parameter_names={"log_scales", "logit_opacities", "quats", "means", "sh0", "shN"},
            reset_adam_step_counts=False,
        )
        return n

    @torch.no_grad()
    def _prune(self, n: int) -> int:
        """
        Remove the ``n`` lowest-opacity Gaussians from the model.

        Ranks all Gaussians by opacity (ascending), builds a boolean keep-mask
        that retains the top ``num_gaussians - n`` entries, and delegates to
        :meth:`filter_gaussians` for the actual state/optimizer update.

        Args:
            n: Number of Gaussians to remove.  Clamped to
               ``num_gaussians - 1`` so at least one Gaussian always survives.

        Returns:
            The number of Gaussians actually removed.
        """
        if n <= 0:
            return 0

        num_gs = self._model.num_gaussians
        n = min(n, num_gs - 1)
        if n <= 0:
            return 0

        _, sorted_indices = torch.sort(self._model.opacities.flatten())
        keep_indices = sorted_indices[n:]
        self.filter_gaussians(keep_indices)

        removed = num_gs - self._model.num_gaussians
        self._logger.info(f"Pruned {removed:,} lowest-opacity Gaussians ({num_gs:,} -> {self._model.num_gaussians:,})")
        return removed

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.__class__.name(),
            "optimizer": self._optimizer.state_dict(),
            "means_lr_decay_exponent": self._means_lr_decay_exponent,
            "config": vars(self._config),
            "spatial_scale": self._spatial_scale,
            "step_count": self._step_count,
            "refine_count": self._refine_count,
            "controller": self._controller.state_dict(),
            "version": 1,
        }

    @torch.no_grad()
    def filter_gaussians(self, indices_or_mask: torch.Tensor):
        def _copy_param_and_grad(param: torch.Tensor) -> torch.Tensor:
            new_param = param[indices_or_mask]
            new_param.grad = param.grad[indices_or_mask] if param.grad is not None else None
            return new_param

        self._model.set_state(
            means=_copy_param_and_grad(self._model.means),
            quats=_copy_param_and_grad(self._model.quats),
            log_scales=_copy_param_and_grad(self._model.log_scales),
            logit_opacities=_copy_param_and_grad(self._model.logit_opacities),
            sh0=_copy_param_and_grad(self._model.sh0),
            shN=_copy_param_and_grad(self._model.shN),
        )
        self._update_optimizer_params_and_state(lambda x: x[indices_or_mask])

    def reset_learning_rates_and_decay(self, batch_size: int, expected_steps: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if expected_steps <= 0:
            raise ValueError("expected_steps must be > 0")

        self._means_lr_decay_exponent = 0.01 ** (1.0 / expected_steps)

        lr_batch_rescale = math.sqrt(float(batch_size))

        reset_lr_values = {
            "means": self._config.means_lr * self._spatial_scale * lr_batch_rescale,
            "log_scales": self._config.log_scales_lr * lr_batch_rescale,
            "quats": self._config.quats_lr * lr_batch_rescale,
            "logit_opacities": self._config.logit_opacities_lr * lr_batch_rescale,
            "sh0": self._config.sh0_lr * lr_batch_rescale,
            "shN": self._config.shN_lr * lr_batch_rescale,
        }

        rescaled_betas = (1.0 - batch_size * (1.0 - 0.9), 1.0 - batch_size * (1.0 - 0.999))
        for param_group in self._optimizer.param_groups:
            param_group["betas"] = rescaled_betas
            param_group["lr"] = reset_lr_values[param_group["name"]]
            param_group["eps"] = 1e-15 / lr_batch_rescale

        self._ensure_means_lr_scheduler()
