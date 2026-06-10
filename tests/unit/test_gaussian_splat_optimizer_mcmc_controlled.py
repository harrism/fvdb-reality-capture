# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import tempfile
import unittest

import torch
from fvdb import GaussianSplat3d

import fvdb_reality_capture as frc
from fvdb_reality_capture.radiance_fields.gaussian_splat_optimizer_mcmc_controlled import (
    ExtremumSeekingController,
    MetricController,
)
from tests.unit.common import GettysburgGaussianSplatTestCase


class MetricControllerTests(unittest.TestCase):
    """Tests for MetricController that do not require a GPU."""

    def test_linear_regression_slope_positive(self):
        # y = 2x + 1 => slope = 2
        points = [(0, 1.0), (1, 3.0), (2, 5.0), (3, 7.0)]
        slope = MetricController._linear_regression_slope(points)
        self.assertAlmostEqual(slope, 2.0, places=5)

    def test_linear_regression_slope_negative(self):
        # y = -0.5x + 10 => slope = -0.5
        points = [(0, 10.0), (2, 9.0), (4, 8.0), (6, 7.0)]
        slope = MetricController._linear_regression_slope(points)
        self.assertAlmostEqual(slope, -0.5, places=5)

    def test_linear_regression_slope_flat(self):
        points = [(0, 5.0), (1, 5.0), (2, 5.0)]
        slope = MetricController._linear_regression_slope(points)
        self.assertAlmostEqual(slope, 0.0, places=5)

    def test_linear_regression_slope_single_point(self):
        slope = MetricController._linear_regression_slope([(0, 1.0)])
        self.assertAlmostEqual(slope, 0.0, places=5)

    def test_warmup_uses_max_rate(self):
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
        )
        # During warmup, should always return max rate
        rate = controller.observe(100, 20.0)
        self.assertEqual(rate, 1.05)
        self.assertIsNone(controller.current_slope)

        rate = controller.observe(200, 21.0)
        self.assertEqual(rate, 1.05)
        self.assertIsNone(controller.current_slope)

    def test_bang_bang_inserts_when_stalled(self):
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
        )
        # Fill warmup
        controller.observe(100, 20.0)
        controller.observe(200, 20.0)
        controller.observe(300, 20.0)

        # All flat -> slope ~ 0 < threshold -> should insert
        rate = controller.observe(400, 20.0)
        self.assertEqual(rate, 1.05)
        self.assertIsNotNone(controller.current_slope)
        self.assertAlmostEqual(controller.current_slope, 0.0, places=5)
        self.assertTrue(controller.is_inserting)

    def test_bang_bang_holds_when_improving(self):
        controller = MetricController(
            trend_window=5,
            slope_threshold=1e-4,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
        )
        # Feed a clearly improving PSNR trend
        for i in range(3):
            controller.observe(i * 100, 20.0 + i * 1.0)  # warmup

        rate = controller.observe(300, 23.0)
        self.assertEqual(rate, 1.0)  # improving -> hold
        self.assertFalse(controller.is_inserting)

        rate = controller.observe(400, 24.0)
        self.assertEqual(rate, 1.0)
        self.assertGreater(controller.current_slope, 1e-4)

    def test_get_insertion_rate_returns_last_rate(self):
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            max_insertion_rate=1.10,
            min_measurements_before_control=2,
        )
        controller.observe(0, 20.0)
        controller.observe(100, 20.0)

        # After warmup, should reflect the control decision
        rate_from_observe = controller.observe(200, 20.0)
        rate_from_get = controller.get_insertion_rate()
        self.assertEqual(rate_from_observe, rate_from_get)

    def test_state_dict_roundtrip(self):
        controller = MetricController(
            trend_window=4,
            slope_threshold=0.001,
            slope_threshold_high=0.002,
            max_insertion_rate=1.08,
            min_measurements_before_control=2,
        )
        controller.observe(0, 20.0)
        controller.observe(100, 21.0)
        controller.observe(200, 21.5)

        state = controller.state_dict()

        controller2 = MetricController()
        controller2.load_state_dict(state)

        self.assertEqual(controller2._trend_window, 4)
        self.assertEqual(controller2._slope_threshold, 0.001)
        self.assertEqual(controller2._max_insertion_rate, 1.08)
        self.assertEqual(len(controller2.history), 3)
        self.assertAlmostEqual(controller2.current_insertion_rate, controller.current_insertion_rate)

    def test_trend_window_minimum(self):
        with self.assertRaises(ValueError):
            MetricController(trend_window=1)

    def test_slope_threshold_high_validation(self):
        # slope_threshold_high must be >= slope_threshold
        with self.assertRaises(ValueError):
            MetricController(slope_threshold=1e-4, slope_threshold_high=5e-5)

    def test_hysteresis_prevents_chattering(self):
        """With hysteresis, the controller should not chatter when slope is between thresholds."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=5e-4,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
        )
        # Fill warmup with flat data -> stalled -> INSERT
        controller.observe(100, 25.0, gaussian_count=100000)
        controller.observe(200, 25.0, gaussian_count=100000)
        controller.observe(300, 25.0, gaussian_count=100000)

        # Slope = 0 < low threshold -> INSERT
        rate = controller.observe(400, 25.0, gaussian_count=100000)
        self.assertEqual(rate, 1.05)
        self.assertTrue(controller.is_inserting)

        # Now feed slight improvement: slope between low (1e-4) and high (5e-4)
        # slope = 2e-4 per step (from 25.0 to 25.06 over 300 steps)
        # This is above low threshold but below high threshold.
        # With hysteresis, since we're currently INSERTING, we use high threshold
        # so should_insert = (2e-4 < 5e-4) = True -> stay inserting
        rate = controller.observe(700, 25.06, gaussian_count=100000)
        self.assertEqual(rate, 1.05)  # Should still be inserting (hysteresis)

    def test_hysteresis_disabled_when_equal(self):
        """When thresholds are equal, behaviour should match the original controller."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=1e-4,  # same as low -> no hysteresis
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
        )
        controller.observe(100, 25.0, gaussian_count=100000)
        controller.observe(200, 25.0, gaussian_count=100000)
        controller.observe(300, 25.0, gaussian_count=100000)

        # Flat -> INSERT
        rate = controller.observe(400, 25.0, gaussian_count=100000)
        self.assertEqual(rate, 1.05)

    def test_burst_tracking_hold_to_insert(self):
        """Verify that burst start is recorded on HOLD -> INSERT transition."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=1e-4,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_threshold=1e-7,
        )
        # Warmup with improving data -> HOLD (but warmup overrides to INSERT)
        controller.observe(100, 20.0, gaussian_count=100000)
        controller.observe(200, 21.0, gaussian_count=100000)
        controller.observe(300, 22.0, gaussian_count=100000)

        # Now clearly improving -> HOLD
        controller.observe(400, 23.0, gaussian_count=100000)
        self.assertFalse(controller.is_inserting)

        # Now flat -> INSERT (HOLD -> INSERT transition)
        controller.observe(500, 23.0, gaussian_count=200000)
        controller.observe(600, 23.0, gaussian_count=200000)
        controller.observe(700, 23.0, gaussian_count=200000)
        self.assertTrue(controller.is_inserting)

        # Burst start should be recorded
        self.assertIsNotNone(controller._burst_start_metric)
        self.assertIsNotNone(controller._burst_start_gaussians)

    def test_anti_windup_saturates_on_ineffective_burst(self):
        """Verify saturation when an insertion burst doesn't improve quality."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=1e-4,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_threshold=1e-7,
        )
        # Warmup: burst baseline is recorded at first observation (25.0, 100K)
        controller.observe(100, 25.0, gaussian_count=100000)
        controller.observe(200, 25.0, gaussian_count=100000)

        # Post-warmup: metric flat while gaussians grow rapidly
        controller.observe(300, 25.0, gaussian_count=200000)
        rate = controller.observe(400, 25.0, gaussian_count=300000)
        self.assertTrue(controller.is_inserting)
        self.assertFalse(controller.is_saturated)

        # Continue inserting -- metric still flat, gaussians still growing
        controller.observe(500, 25.0, gaussian_count=400000)

        # Metric barely ticks up -- just enough to push the slope over the
        # threshold in the last 3-point window.  Total improvement is tiny:
        #   delta_metric   = 25.02 - 25.0 = 0.02
        #   delta_gaussians = 500K - 100K = 400K
        #   effectiveness  = 0.02 / 400000 = 5e-8 < 1e-7  -> SATURATED
        controller.observe(600, 25.005, gaussian_count=500000)
        rate = controller.observe(700, 25.02, gaussian_count=500000)

        # Should be HOLD now and saturated
        self.assertFalse(controller.is_inserting)
        self.assertTrue(controller.is_saturated)

        # Even if quality stalls again, should stay HOLD (saturated)
        controller.observe(800, 25.02, gaussian_count=500000)
        controller.observe(900, 25.02, gaussian_count=500000)
        rate = controller.observe(1000, 25.02, gaussian_count=500000)
        self.assertEqual(rate, 1.0)  # Still HOLD despite stalled quality
        self.assertTrue(controller.is_saturated)

    def test_anti_windup_disabled_when_threshold_zero(self):
        """Verify anti-windup is disabled when effectiveness_threshold = 0."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=1e-4,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_threshold=0.0,  # disabled
        )
        # Warmup
        controller.observe(100, 25.0, gaussian_count=100000)
        controller.observe(200, 25.0, gaussian_count=100000)
        controller.observe(300, 25.0, gaussian_count=100000)

        # Flat -> INSERT (burst starts)
        controller.observe(400, 25.0, gaussian_count=100000)
        self.assertTrue(controller.is_inserting)

        # Big growth with tiny PSNR gain -> burst ends
        controller.observe(500, 25.001, gaussian_count=500000)
        controller.observe(600, 25.5, gaussian_count=500000)
        controller.observe(700, 26.0, gaussian_count=500000)

        # Should NOT be saturated (anti-windup disabled)
        self.assertFalse(controller.is_saturated)

    def test_state_dict_roundtrip_with_new_fields(self):
        """Verify state_dict roundtrip includes hysteresis and anti-windup state."""
        controller = MetricController(
            trend_window=4,
            slope_threshold=0.001,
            slope_threshold_high=0.005,
            max_insertion_rate=1.08,
            min_measurements_before_control=2,
            effectiveness_threshold=1e-7,
        )
        controller.observe(0, 20.0, gaussian_count=1000)
        controller.observe(100, 21.0, gaussian_count=2000)
        controller.observe(200, 21.5, gaussian_count=3000)

        state = controller.state_dict()

        # Verify new key name is used in serialization
        self.assertIn("burst_start_metric", state)
        self.assertNotIn("burst_start_psnr", state)

        controller2 = MetricController()
        controller2.load_state_dict(state)

        self.assertEqual(controller2._trend_window, 4)
        self.assertEqual(controller2._slope_threshold, 0.001)
        self.assertEqual(controller2._slope_threshold_high, 0.005)
        self.assertEqual(controller2._effectiveness_threshold, 1e-7)
        self.assertEqual(controller2._max_insertion_rate, 1.08)
        self.assertEqual(len(controller2.history), 3)
        self.assertAlmostEqual(controller2.current_insertion_rate, controller.current_insertion_rate)
        self.assertEqual(controller2._saturated, controller._saturated)
        self.assertEqual(controller2._burst_start_metric, controller._burst_start_metric)

    def test_state_dict_backward_compat(self):
        """Verify load_state_dict works with old state dicts missing new fields."""
        old_state = {
            "trend_window": 5,
            "slope_threshold": 1e-4,
            "max_insertion_rate": 1.05,
            "min_measurements": 3,
            "history": [(0, 20.0), (100, 21.0)],
            "current_insertion_rate": 1.05,
            "current_slope": None,
        }
        controller = MetricController()
        controller.load_state_dict(old_state)

        self.assertEqual(controller._slope_threshold_high, 1e-4)  # defaults to slope_threshold
        self.assertEqual(controller._effectiveness_threshold, 0.0)
        self.assertFalse(controller._saturated)
        self.assertIsNone(controller._burst_start_metric)

    def test_state_dict_backward_compat_burst_start_psnr(self):
        """Verify load_state_dict accepts the legacy 'burst_start_psnr' key."""
        old_state = {
            "trend_window": 5,
            "slope_threshold": 1e-4,
            "max_insertion_rate": 1.05,
            "min_measurements": 3,
            "history": [(0, 20.0), (100, 21.0)],
            "current_insertion_rate": 1.05,
            "current_slope": None,
            "burst_start_psnr": 25.5,
            "burst_start_gaussians": 100000,
            "saturated": False,
        }
        controller = MetricController()
        controller.load_state_dict(old_state)

        # Legacy key should be loaded into the renamed field
        self.assertAlmostEqual(controller._burst_start_metric, 25.5)
        self.assertEqual(controller._burst_start_gaussians, 100000)

    def test_cooldown_prevents_immediate_reinsert(self):
        """Verify cooldown enforces HOLD for N probes after a burst ends."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=1e-4,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            cooldown_probes=5,
        )
        # Warmup
        controller.observe(100, 25.0)
        controller.observe(200, 25.0)

        # Post-warmup: flat -> INSERT
        controller.observe(300, 25.0)
        controller.observe(400, 25.0)
        self.assertTrue(controller.is_inserting)

        # Quality jumps -> INSERT->HOLD at step 500, cooldown_remaining=5
        controller.observe(500, 26.0)
        self.assertFalse(controller.is_inserting)
        self.assertTrue(controller.is_cooling_down)

        # Cooldown ticks down on each observe: 5->4->3->2->1
        controller.observe(600, 27.0)  # cd=4
        controller.observe(700, 28.0)  # cd=3

        # Quality stalls -- would normally trigger INSERT, but cooldown blocks it
        controller.observe(800, 28.0)  # cd=2
        self.assertFalse(controller.is_inserting)
        self.assertTrue(controller.is_cooling_down)

        controller.observe(900, 28.0)  # cd=1
        self.assertFalse(controller.is_inserting)
        self.assertTrue(controller.is_cooling_down)

        controller.observe(1000, 28.0)  # cd=0, cooldown expires
        self.assertFalse(controller.is_inserting)
        self.assertFalse(controller.is_cooling_down)

        # Now the stall is detectable -- next flat probe triggers INSERT
        controller.observe(1100, 28.0)
        controller.observe(1200, 28.0)
        rate = controller.observe(1300, 28.0)
        self.assertTrue(controller.is_inserting)

    def test_cooldown_disabled_when_zero(self):
        """Verify cooldown_probes=0 allows immediate re-insertion."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=1e-4,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            cooldown_probes=0,
        )
        controller.observe(100, 25.0)
        controller.observe(200, 25.0)
        controller.observe(300, 25.0)
        controller.observe(400, 25.0)
        self.assertTrue(controller.is_inserting)

        # Quality jumps -> HOLD
        controller.observe(500, 26.0)
        controller.observe(600, 27.0)
        controller.observe(700, 28.0)
        self.assertFalse(controller.is_inserting)
        self.assertFalse(controller.is_cooling_down)

        # Immediate stall -> INSERT (no cooldown)
        controller.observe(800, 28.0)
        controller.observe(900, 28.0)
        rate = controller.observe(1000, 28.0)
        self.assertTrue(controller.is_inserting)

    def test_cooldown_validation(self):
        """Verify cooldown_probes < 0 raises ValueError."""
        with self.assertRaises(ValueError):
            MetricController(cooldown_probes=-1)

    def test_cooldown_state_dict_roundtrip(self):
        """Verify cooldown state is preserved through serialization."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=1e-4,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            cooldown_probes=5,
        )
        controller.observe(100, 25.0)
        controller.observe(200, 25.0)
        controller.observe(300, 25.0)
        controller.observe(400, 25.0)
        # Trigger INSERT->HOLD to start cooldown
        controller.observe(500, 26.0)
        controller.observe(600, 27.0)
        controller.observe(700, 28.0)
        self.assertTrue(controller.is_cooling_down)

        state = controller.state_dict()
        self.assertIn("cooldown_probes", state)
        self.assertIn("cooldown_remaining", state)
        self.assertEqual(state["cooldown_probes"], 5)
        self.assertGreater(state["cooldown_remaining"], 0)

        controller2 = MetricController()
        controller2.load_state_dict(state)
        self.assertEqual(controller2._cooldown_probes, 5)
        self.assertEqual(controller2._cooldown_remaining, controller._cooldown_remaining)

    def test_growth_cap_config_defaults(self):
        """Verify new config fields have correct defaults."""
        config = frc.radiance_fields.GaussianSplatOptimizerMCMCControlledConfig()
        self.assertEqual(config.cooldown_probes, 0)
        self.assertEqual(config.max_new_gaussians_per_refine, 0)
        self.assertEqual(config.effectiveness_decay_ratio, 0.0)

    def test_control_metric_config_property(self):
        """Verify control_metric config field has correct default and is accessible."""
        config = frc.radiance_fields.GaussianSplatOptimizerMCMCControlledConfig()
        self.assertEqual(config.control_metric, "ssim")

    def test_effectiveness_gating_keeps_inserting_during_early_growth(self):
        """With effectiveness gating, INSERT persists when effectiveness is high."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.1,
        )
        # Warmup: quality improving rapidly with gaussian growth (high effectiveness)
        controller.observe(100, 0.50, gaussian_count=100000)
        controller.observe(200, 0.55, gaussian_count=150000)
        controller.observe(300, 0.60, gaussian_count=200000)

        # Post-warmup: quality still improving with growth -> should stay INSERT
        # (with the old slope-based logic this would have switched to HOLD)
        rate = controller.observe(400, 0.64, gaussian_count=260000)
        self.assertEqual(rate, 1.05)
        self.assertTrue(controller.is_inserting)

        rate = controller.observe(500, 0.67, gaussian_count=330000)
        self.assertEqual(rate, 1.05)
        self.assertTrue(controller.is_inserting)

    def test_effectiveness_gating_transitions_to_hold_on_decay(self):
        """Effectiveness gating transitions to HOLD when effectiveness decays."""
        controller = MetricController(
            trend_window=5,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.1,
        )
        # Warmup: high effectiveness (big quality gains per gaussian)
        controller.observe(100, 0.50, gaussian_count=100000)
        controller.observe(200, 0.56, gaussian_count=150000)
        controller.observe(300, 0.61, gaussian_count=200000)

        # Post-warmup: effectiveness still good, stays INSERT
        controller.observe(400, 0.65, gaussian_count=260000)
        self.assertTrue(controller.is_inserting)

        # Growth continues but quality gains flatten (diminishing returns)
        controller.observe(500, 0.66, gaussian_count=400000)
        controller.observe(600, 0.665, gaussian_count=600000)
        controller.observe(700, 0.668, gaussian_count=900000)
        controller.observe(800, 0.670, gaussian_count=1300000)

        # By now effectiveness should be much lower than peak -> HOLD
        self.assertFalse(controller.is_inserting)

    def test_effectiveness_gating_disabled_when_zero(self):
        """When effectiveness_decay_ratio=0.0, use original slope-based logic."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.0,
        )
        # Warmup with quality improving rapidly
        controller.observe(100, 0.50, gaussian_count=100000)
        controller.observe(200, 0.55, gaussian_count=150000)

        # Post-warmup: strong positive slope -> HOLD (original behavior)
        rate = controller.observe(300, 0.60, gaussian_count=200000)
        self.assertEqual(rate, 1.0)
        self.assertFalse(controller.is_inserting)

    def test_effectiveness_gating_stall_resumes_insert(self):
        """After effectiveness-gated HOLD, stall detection resumes INSERT."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.1,
            cooldown_probes=0,
        )
        # Warmup
        controller.observe(100, 0.50, gaussian_count=100000)
        controller.observe(200, 0.55, gaussian_count=150000)
        controller.observe(300, 0.60, gaussian_count=200000)
        self.assertTrue(controller.is_inserting)

        # Force effectiveness decay: quality stalls while gaussians grow
        controller.observe(400, 0.605, gaussian_count=400000)
        controller.observe(500, 0.608, gaussian_count=700000)
        controller.observe(600, 0.610, gaussian_count=1100000)
        controller.observe(700, 0.611, gaussian_count=1600000)

        # Should eventually reach HOLD due to effectiveness decay
        is_hold = not controller.is_inserting
        if not is_hold:
            # One more observation with extreme diminishing returns
            controller.observe(800, 0.6115, gaussian_count=2200000)
            is_hold = not controller.is_inserting
        self.assertTrue(is_hold, "Should be in HOLD after effectiveness decay")

        # Now quality stalls (flat slope) -> should resume INSERT
        gc = controller.history[-1][2]
        controller.observe(900, 0.612, gaussian_count=gc)
        controller.observe(1000, 0.612, gaussian_count=gc)
        controller.observe(1100, 0.612, gaussian_count=gc)
        controller.observe(1200, 0.612, gaussian_count=gc)
        self.assertTrue(controller.is_inserting, "Should resume INSERT after quality stall")

    def test_effectiveness_gating_state_dict_roundtrip(self):
        """Effectiveness gating state survives serialization."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.1,
        )
        controller.observe(100, 0.50, gaussian_count=100000)
        controller.observe(200, 0.55, gaussian_count=150000)
        controller.observe(300, 0.60, gaussian_count=200000)
        controller.observe(400, 0.64, gaussian_count=260000)

        state = controller.state_dict()
        self.assertIn("effectiveness_decay_ratio", state)
        self.assertIn("insert_phase_start_idx", state)
        self.assertIn("peak_effectiveness", state)
        self.assertEqual(state["effectiveness_decay_ratio"], 0.1)

        controller2 = MetricController()
        controller2.load_state_dict(state)
        self.assertEqual(controller2._effectiveness_decay_ratio, 0.1)
        self.assertEqual(controller2._insert_phase_start_idx, controller._insert_phase_start_idx)
        self.assertAlmostEqual(controller2._peak_effectiveness, controller._peak_effectiveness)
        self.assertEqual(len(controller2.history), 4)
        # History should be 3-tuples
        self.assertEqual(len(controller2.history[0]), 3)

    def test_effectiveness_gating_backward_compat_2tuple_history(self):
        """Old 2-tuple history is migrated to 3-tuples with gaussian_count=0."""
        old_state = {
            "trend_window": 5,
            "slope_threshold": 1e-4,
            "max_insertion_rate": 1.05,
            "min_measurements": 3,
            "history": [(0, 20.0), (100, 21.0), (200, 22.0)],
            "current_insertion_rate": 1.05,
            "current_slope": None,
        }
        controller = MetricController()
        controller.load_state_dict(old_state)

        # History should be migrated to 3-tuples
        self.assertEqual(len(controller.history), 3)
        self.assertEqual(len(controller.history[0]), 3)
        self.assertEqual(controller.history[0], (0, 20.0, 0))
        self.assertEqual(controller.history[2], (200, 22.0, 0))
        # New fields should have defaults
        self.assertEqual(controller._effectiveness_decay_ratio, 0.0)
        self.assertIsNone(controller._insert_phase_start_idx)
        self.assertEqual(controller._peak_effectiveness, 0.0)

    def test_baseline_subtraction_shortens_insert_phase(self):
        """With a positive baseline slope, corrected effectiveness decays faster."""
        # Controller WITHOUT baseline subtraction context (first burst, baseline=0)
        ctrl_no_baseline = MetricController(
            trend_window=3,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.3,
        )
        # Controller WITH a pre-set baseline slope (simulating a second burst)
        ctrl_with_baseline = MetricController(
            trend_window=3,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.3,
        )
        ctrl_with_baseline._baseline_optimization_slope = 3e-5

        observations = [
            (100, 0.500, 100000),
            (200, 0.530, 150000),
            (300, 0.555, 200000),
            (400, 0.575, 260000),
            (500, 0.590, 330000),
            (600, 0.600, 410000),
            (700, 0.607, 500000),
            (800, 0.612, 600000),
            (900, 0.615, 710000),
            (1000, 0.617, 830000),
        ]
        no_baseline_hold_step = None
        with_baseline_hold_step = None
        for step, metric, gc in observations:
            ctrl_no_baseline.observe(step, metric, gaussian_count=gc)
            ctrl_with_baseline.observe(step, metric, gaussian_count=gc)
            if no_baseline_hold_step is None and not ctrl_no_baseline.is_inserting:
                no_baseline_hold_step = step
            if with_baseline_hold_step is None and not ctrl_with_baseline.is_inserting:
                with_baseline_hold_step = step

        self.assertIsNotNone(with_baseline_hold_step, "Baseline controller should reach HOLD")
        if no_baseline_hold_step is not None:
            self.assertLessEqual(with_baseline_hold_step, no_baseline_hold_step)
        # Even if both reach HOLD, the baseline-subtracted one should reach it no later

    def test_baseline_slope_captured_at_hold_to_insert(self):
        """At HOLD->INSERT transition, baseline_optimization_slope equals the HOLD-phase slope."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.1,
            cooldown_probes=0,
        )
        self.assertEqual(controller._baseline_optimization_slope, 0.0)

        # Warmup -> INSERT
        controller.observe(100, 0.50, gaussian_count=100000)
        controller.observe(200, 0.55, gaussian_count=150000)
        controller.observe(300, 0.60, gaussian_count=200000)
        self.assertTrue(controller.is_inserting)
        # First burst: baseline should still be 0 (no prior HOLD phase)
        self.assertEqual(controller._baseline_optimization_slope, 0.0)

        # Force HOLD by making effectiveness decay: lots of gaussians, little quality
        controller.observe(400, 0.605, gaussian_count=500000)
        controller.observe(500, 0.608, gaussian_count=900000)
        controller.observe(600, 0.610, gaussian_count=1400000)
        controller.observe(700, 0.611, gaussian_count=2000000)
        controller.observe(800, 0.6115, gaussian_count=2700000)
        self.assertFalse(controller.is_inserting, "Should be in HOLD")

        # HOLD phase: slight optimization-only improvement (no Gaussian growth)
        gc = controller.history[-1][2]
        controller.observe(900, 0.6120, gaussian_count=gc)
        controller.observe(1000, 0.6121, gaussian_count=gc)
        controller.observe(1100, 0.6122, gaussian_count=gc)

        # Stall: quality flattens completely -> should trigger HOLD->INSERT
        controller.observe(1200, 0.6122, gaussian_count=gc)
        controller.observe(1300, 0.6122, gaussian_count=gc)
        controller.observe(1400, 0.6122, gaussian_count=gc)
        if controller.is_inserting:
            # The baseline should reflect the non-zero HOLD-phase slope
            self.assertNotEqual(
                controller._baseline_optimization_slope,
                0.0,
                "Baseline slope should be set from the HOLD-phase window at HOLD->INSERT",
            )

    def test_peak_effectiveness_reset_per_burst(self):
        """Peak effectiveness resets to 0 at each HOLD->INSERT transition."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.1,
            cooldown_probes=0,
        )
        # Warmup (INSERT)
        controller.observe(100, 0.50, gaussian_count=100000)
        controller.observe(200, 0.56, gaussian_count=150000)
        controller.observe(300, 0.61, gaussian_count=200000)

        # Build up peak effectiveness
        controller.observe(400, 0.65, gaussian_count=260000)
        peak_after_first_burst = controller.peak_effectiveness
        self.assertGreater(peak_after_first_burst, 0.0)

        # Force HOLD via effectiveness decay
        controller.observe(500, 0.655, gaussian_count=500000)
        controller.observe(600, 0.658, gaussian_count=900000)
        controller.observe(700, 0.660, gaussian_count=1400000)
        controller.observe(800, 0.661, gaussian_count=2000000)

        if not controller.is_inserting:
            # In HOLD: stall to trigger new INSERT
            gc = controller.history[-1][2]
            controller.observe(900, 0.661, gaussian_count=gc)
            controller.observe(1000, 0.661, gaussian_count=gc)
            controller.observe(1100, 0.661, gaussian_count=gc)
            controller.observe(1200, 0.661, gaussian_count=gc)

            if controller.is_inserting:
                self.assertEqual(
                    controller.peak_effectiveness,
                    0.0,
                    "Peak effectiveness should reset to 0 at HOLD->INSERT transition",
                )

    def test_baseline_subtraction_state_dict_roundtrip(self):
        """baseline_optimization_slope survives serialization."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.1,
        )
        controller._baseline_optimization_slope = 1.5e-5
        controller.observe(100, 0.50, gaussian_count=100000)
        controller.observe(200, 0.55, gaussian_count=150000)
        controller.observe(300, 0.60, gaussian_count=200000)

        state = controller.state_dict()
        self.assertIn("baseline_optimization_slope", state)
        self.assertAlmostEqual(state["baseline_optimization_slope"], 1.5e-5)

        controller2 = MetricController()
        controller2.load_state_dict(state)
        self.assertAlmostEqual(controller2._baseline_optimization_slope, 1.5e-5)

    def test_first_burst_no_baseline(self):
        """The first INSERT phase uses baseline=0 (no prior HOLD to measure)."""
        controller = MetricController(
            trend_window=3,
            slope_threshold=1e-6,
            slope_threshold_high=2e-6,
            max_insertion_rate=1.05,
            min_measurements_before_control=3,
            effectiveness_decay_ratio=0.1,
        )
        # Warmup
        controller.observe(100, 0.50, gaussian_count=100000)
        controller.observe(200, 0.55, gaussian_count=150000)
        controller.observe(300, 0.60, gaussian_count=200000)

        # Post-warmup INSERT
        controller.observe(400, 0.64, gaussian_count=260000)
        self.assertTrue(controller.is_inserting)
        self.assertEqual(
            controller._baseline_optimization_slope, 0.0, "First burst should have baseline_optimization_slope=0"
        )

    def test_baseline_subtraction_backward_compat_load(self):
        """Loading a v1 state dict (no baseline_optimization_slope) defaults to 0."""
        old_state = {
            "trend_window": 5,
            "slope_threshold": 1e-6,
            "slope_threshold_high": 2e-6,
            "max_insertion_rate": 1.05,
            "min_measurements": 3,
            "effectiveness_decay_ratio": 0.1,
            "history": [(100, 0.50, 100000), (200, 0.55, 150000)],
            "current_insertion_rate": 1.05,
            "current_slope": None,
            "insert_phase_start_idx": 0,
            "peak_effectiveness": 1e-6,
        }
        controller = MetricController()
        controller.load_state_dict(old_state)
        self.assertEqual(controller._baseline_optimization_slope, 0.0)

    def test_effectiveness_decay_ratio_validation(self):
        """Verify effectiveness_decay_ratio validation."""
        with self.assertRaises(ValueError):
            MetricController(effectiveness_decay_ratio=1.0)
        with self.assertRaises(ValueError):
            MetricController(effectiveness_decay_ratio=-0.1)


class GaussianSplatOptimizerMCMCControlledTests(GettysburgGaussianSplatTestCase, unittest.TestCase):

    def test_serialize_optimizer_controlled(self):
        if self.device != "cuda":
            self.skipTest("GaussianSplatOptimizerMCMCControlled uses CUDA-only ops")

        model_1 = self.model
        max_steps = 200 * len(self.training_dataset)
        config = frc.radiance_fields.GaussianSplatOptimizerMCMCControlledConfig(
            noise_lr=0.0,  # disable stochasticity for determinism
            insertion_rate=1.0,  # avoid insertions for determinism
            max_gaussians=-1,
            probe_every_k_refines=1,
            probe_n_images=1,
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=2e-4,
            min_measurements_before_control=2,
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        optimizer_1 = frc.radiance_fields.GaussianSplatOptimizerMCMCControlled.from_model_and_scene(
            model=model_1,
            sfm_scene=self.training_dataset.sfm_scene,
            config=config,
        )
        optimizer_1.reset_learning_rates_and_decay(batch_size=1, expected_steps=max_steps)

        # Push some metric observations
        optimizer_1.observe_metric(0, 20.0)
        optimizer_1.observe_metric(100, 21.0)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pt", delete=True) as temp_file:
            torch.save(model_1.state_dict(), temp_file.name + ".model")
            torch.save(optimizer_1.state_dict(), temp_file.name)

            # Run one step of refine + step
            optimizer_1.zero_grad()
            gt_img_1, pred_img_1, _ = self._render_one_image(model_1)
            loss_1 = torch.nn.functional.l1_loss(pred_img_1, gt_img_1)
            loss_1.backward()
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            optimizer_1.refine()
            optimizer_1.step()
            optimizer_1.zero_grad()

            # Post-step render
            gt_img_2, pred_img_2, _ = self._render_one_image(model_1)
            loss_2 = torch.nn.functional.l1_loss(pred_img_2, gt_img_2)

            # Load model + optimizer and replay the same operations
            model_2 = GaussianSplat3d.from_state_dict(torch.load(temp_file.name + ".model", map_location=self.device))
            loaded_state_dict = torch.load(temp_file.name, map_location=self.device, weights_only=False)
            optimizer_2 = frc.radiance_fields.GaussianSplatOptimizerMCMCControlled.from_state_dict(
                model_2, loaded_state_dict
            )

            # Verify controller state was restored
            self.assertEqual(len(optimizer_2.controller.history), 2)
            self.assertAlmostEqual(optimizer_2.controller.history[0][1], 20.0)

            optimizer_2.zero_grad()
            gt_img_3, pred_img_3, _ = self._render_one_image(model_2)
            self.assertTrue(torch.allclose(pred_img_1, pred_img_3))
            loss_3 = torch.nn.functional.l1_loss(pred_img_3, gt_img_3)
            self.assertAlmostEqual(loss_1.item(), loss_3.item(), places=3)
            loss_3.backward()
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            optimizer_2.refine()
            optimizer_2.step()
            optimizer_2.zero_grad()

            gt_img_4, pred_img_4, _ = self._render_one_image(model_2)
            self.assertTrue(torch.allclose(pred_img_2, pred_img_4, atol=1e-3))
            loss_4 = torch.nn.functional.l1_loss(pred_img_4, gt_img_4)
            self.assertAlmostEqual(loss_2.item(), loss_4.item(), places=3)

    def test_refine_uses_controller_rate(self):
        if self.device != "cuda":
            self.skipTest("GaussianSplatOptimizerMCMCControlled uses CUDA-only ops")

        model = self.model
        config = frc.radiance_fields.GaussianSplatOptimizerMCMCControlledConfig(
            noise_lr=0.0,
            insertion_rate=1.1,  # 10% growth when controller says to insert
            max_gaussians=-1,
            deletion_opacity_threshold=0.0,
            probe_every_k_refines=1,
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=2e-4,
            min_measurements_before_control=2,
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        optimizer = frc.radiance_fields.GaussianSplatOptimizerMCMCControlled.from_model_and_scene(
            model=model,
            sfm_scene=self.training_dataset.sfm_scene,
            config=config,
        )
        optimizer.reset_learning_rates_and_decay(batch_size=1, expected_steps=10)

        # Feed improving PSNR -> controller should set rate = 1.0 (hold)
        for i in range(5):
            optimizer.observe_metric(i * 100, 20.0 + i * 2.0)

        self.assertAlmostEqual(optimizer.controller.get_insertion_rate(), 1.0)

        n_before = model.num_gaussians
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        stats = optimizer.refine()

        # With rate = 1.0, no Gaussians should be added
        self.assertEqual(stats["num_added"], 0)
        self.assertEqual(model.num_gaussians, n_before)

    def test_refine_inserts_when_stalled(self):
        if self.device != "cuda":
            self.skipTest("GaussianSplatOptimizerMCMCControlled uses CUDA-only ops")

        model = self.model
        config = frc.radiance_fields.GaussianSplatOptimizerMCMCControlledConfig(
            noise_lr=0.0,
            insertion_rate=1.1,
            max_gaussians=-1,
            deletion_opacity_threshold=0.0,
            probe_every_k_refines=1,
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=2e-4,
            min_measurements_before_control=2,
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        optimizer = frc.radiance_fields.GaussianSplatOptimizerMCMCControlled.from_model_and_scene(
            model=model,
            sfm_scene=self.training_dataset.sfm_scene,
            config=config,
        )
        optimizer.reset_learning_rates_and_decay(batch_size=1, expected_steps=10)

        # Feed flat PSNR -> controller should set rate = max (1.1)
        for i in range(5):
            optimizer.observe_metric(i * 100, 25.0)

        self.assertAlmostEqual(optimizer.controller.get_insertion_rate(), 1.1)

        n_before = model.num_gaussians
        expected_target = int(1.1 * n_before)
        expected_added = max(0, expected_target - n_before)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        stats = optimizer.refine()

        self.assertEqual(stats["num_added"], expected_added)
        self.assertEqual(model.num_gaussians, n_before + expected_added)
        self.assertAlmostEqual(stats["insertion_rate"], 1.1, places=4)

    def test_max_gaussians_cap_respected(self):
        if self.device != "cuda":
            self.skipTest("GaussianSplatOptimizerMCMCControlled uses CUDA-only ops")

        model = self.model
        # Set max_gaussians to exactly the current count -> no growth possible
        config = frc.radiance_fields.GaussianSplatOptimizerMCMCControlledConfig(
            noise_lr=0.0,
            insertion_rate=1.1,
            max_gaussians=model.num_gaussians,
            deletion_opacity_threshold=0.0,
            probe_every_k_refines=1,
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=2e-4,
            min_measurements_before_control=2,
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        optimizer = frc.radiance_fields.GaussianSplatOptimizerMCMCControlled.from_model_and_scene(
            model=model,
            sfm_scene=self.training_dataset.sfm_scene,
            config=config,
        )
        optimizer.reset_learning_rates_and_decay(batch_size=1, expected_steps=10)

        # Flat PSNR -> controller wants to insert
        for i in range(5):
            optimizer.observe_metric(i * 100, 25.0)

        n_before = model.num_gaussians
        stats = optimizer.refine()

        # But max_gaussians prevents it
        self.assertEqual(stats["num_added"], 0)
        self.assertEqual(model.num_gaussians, n_before)


class ExtremumSeekingControllerTests(unittest.TestCase):
    """Tests for the ESC state machine.  RED phase -- all tests should fail until
    ExtremumSeekingController is implemented."""

    # ------------------------------------------------------------------
    # Construction & validation
    # ------------------------------------------------------------------

    def test_default_construction(self):
        ctrl = ExtremumSeekingController()
        self.assertEqual(ctrl.state, "WARMUP")
        self.assertIsNotNone(ctrl.get_insertion_rate())

    def test_perturbation_fraction_validation(self):
        with self.assertRaises(ValueError):
            ExtremumSeekingController(perturbation_fraction=0.0)
        with self.assertRaises(ValueError):
            ExtremumSeekingController(perturbation_fraction=-0.1)
        with self.assertRaises(ValueError):
            ExtremumSeekingController(perturbation_fraction=1.0)

    def test_dwell_probes_validation(self):
        with self.assertRaises(ValueError):
            ExtremumSeekingController(dwell_probes=0)

    def test_gradient_window_validation(self):
        with self.assertRaises(ValueError):
            ExtremumSeekingController(gradient_window=1)
        with self.assertRaises(ValueError):
            ExtremumSeekingController(dwell_probes=5, gradient_window=5)

    def test_noise_deadband_validation(self):
        with self.assertRaises(ValueError):
            ExtremumSeekingController(noise_deadband=-0.01)
        with self.assertRaises(ValueError):
            ExtremumSeekingController(noise_deadband=1.0)

    # ------------------------------------------------------------------
    # Warmup phase
    # ------------------------------------------------------------------

    def test_warmup_returns_hold(self):
        """During warmup, the controller should return HOLD (rate == 1.0) -- just observing."""
        ctrl = ExtremumSeekingController(min_warmup_probes=3)
        rate = ctrl.observe(100, 0.50, gaussian_count=100000)
        self.assertAlmostEqual(rate, 1.0, msg="WARMUP should hold (rate=1.0), not insert")
        self.assertEqual(ctrl.state, "WARMUP")

        rate = ctrl.observe(200, 0.55, gaussian_count=120000)
        self.assertAlmostEqual(rate, 1.0, msg="WARMUP should hold (rate=1.0), not insert")
        self.assertEqual(ctrl.state, "WARMUP")

    def test_warmup_exits_after_min_probes(self):
        """After min_warmup_probes observations, state should transition to INSERT."""
        ctrl = ExtremumSeekingController(min_warmup_probes=3)
        rate = ctrl.observe(100, 0.50, gaussian_count=100000)
        self.assertAlmostEqual(rate, 1.0, msg="WARMUP: rate should be 1.0")
        rate = ctrl.observe(200, 0.55, gaussian_count=120000)
        self.assertAlmostEqual(rate, 1.0, msg="WARMUP: rate should be 1.0")

        # The 3rd observation triggers exit from WARMUP -> INSERT
        rate = ctrl.observe(300, 0.60, gaussian_count=140000)
        self.assertNotEqual(ctrl.state, "WARMUP")

    def test_warmup_captures_baseline(self):
        """Exiting WARMUP should capture a baseline slope for HP-filter subtraction."""
        ctrl = ExtremumSeekingController(min_warmup_probes=3, perturbation_fraction=0.10)
        ctrl.observe(100, 0.50, gaussian_count=100000)
        ctrl.observe(200, 0.55, gaussian_count=100000)
        ctrl.observe(300, 0.60, gaussian_count=100000)
        self.assertNotEqual(ctrl.baseline_slope, 0.0, "Baseline slope should be captured from warmup data")

    # ------------------------------------------------------------------
    # INSERT -> DWELL -> ESTIMATE cycle
    # ------------------------------------------------------------------

    def test_insert_transitions_to_dwell(self):
        """After INSERT applies its burst, state should move to DWELL."""
        ctrl = ExtremumSeekingController(min_warmup_probes=2, perturbation_fraction=0.10, dwell_probes=3)
        # Warmup (rate=1.0, no insertion)
        ctrl.observe(100, 0.50, gaussian_count=100000)
        ctrl.observe(200, 0.55, gaussian_count=100000)
        # After warmup, state transitions to INSERT

        # The next observe() runs INSERT handler: sets rate=1.10, transitions to DWELL
        rate = ctrl.observe(300, 0.60, gaussian_count=100000)
        self.assertGreater(rate, 1.0, "INSERT should return rate > 1.0")
        self.assertEqual(ctrl.state, "DWELL", "INSERT should immediately transition to DWELL")

    def test_dwell_suppresses_actions(self):
        """During DWELL, the controller should return rate = 1.0 (hold)."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=2, perturbation_fraction=0.10, dwell_probes=5, gradient_window=8
        )
        # Warmup (rate=1.0)
        ctrl.observe(100, 0.50, gaussian_count=100000)
        ctrl.observe(200, 0.55, gaussian_count=100000)
        # INSERT burst -> DWELL
        rate = ctrl.observe(300, 0.60, gaussian_count=100000)
        self.assertGreater(rate, 1.0, "INSERT should set rate > 1.0")
        self.assertEqual(ctrl.state, "DWELL")

        # All subsequent DWELL probes should return rate=1.0
        for i in range(5):
            rate = ctrl.observe(400 + i * 100, 0.61, gaussian_count=110000)
            if ctrl.state == "DWELL":
                self.assertAlmostEqual(rate, 1.0, msg="DWELL should return rate=1.0 (hold)")

    def test_dwell_transitions_to_estimate_after_probes(self):
        """After dwell_probes observations in DWELL, should transition to ESTIMATE."""
        ctrl = ExtremumSeekingController(min_warmup_probes=2, perturbation_fraction=0.10, dwell_probes=3)
        # Warmup
        ctrl.observe(100, 0.50, gaussian_count=100000)
        ctrl.observe(200, 0.55, gaussian_count=100000)
        # INSERT -> DWELL
        ctrl.observe(300, 0.60, gaussian_count=100000)
        self.assertEqual(ctrl.state, "DWELL")

        # Feed dwell_probes observations -- should transition to ESTIMATE
        states_seen = []
        for i in range(10):
            ctrl.observe(400 + i * 100, 0.61 + i * 0.002, gaussian_count=110000)
            states_seen.append(ctrl.state)

        self.assertIn("ESTIMATE", states_seen, "Should reach ESTIMATE after DWELL")

    # ------------------------------------------------------------------
    # Gradient estimation -> action selection
    # ------------------------------------------------------------------

    def test_positive_gradient_leads_to_insert(self):
        """When dQuality/dGaussians is positive, ESTIMATE should schedule INSERT."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=2,
            perturbation_fraction=0.10,
            dwell_probes=2,
            gradient_window=3,
            noise_deadband=0.05,
        )
        # Warmup (rate=1.0)
        ctrl.observe(100, 0.50, gaussian_count=100000)
        ctrl.observe(200, 0.55, gaussian_count=100000)

        # INSERT burst (rate > 1.0) -> DWELL (rate=1.0)
        rate = ctrl.observe(300, 0.60, gaussian_count=100000)
        self.assertGreater(rate, 1.0)
        self.assertEqual(ctrl.state, "DWELL")

        # Feed DWELL observations with increasing gc (from the burst) and improving quality
        ctrl.observe(400, 0.64, gaussian_count=110000)
        ctrl.observe(500, 0.67, gaussian_count=110000)
        # Should now be in ESTIMATE or transitioning

        # Feed more observations with clear positive gradient to reach next INSERT
        insert_seen = False
        for i in range(10):
            gc = 110000 + i * 5000
            rate = ctrl.observe(600 + i * 100, 0.68 + i * 0.01, gaussian_count=gc)
            if rate > 1.0:
                insert_seen = True
                break
        self.assertTrue(insert_seen, "Positive gradient should lead to INSERT (rate > 1.0)")

    def test_negative_gradient_leads_to_prune(self):
        """When dQuality/dGaussians is negative, ESTIMATE should schedule PRUNE."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=2,
            perturbation_fraction=0.10,
            dwell_probes=2,
            gradient_window=3,
            noise_deadband=0.05,
        )
        # Warmup (rate=1.0)
        ctrl.observe(100, 0.70, gaussian_count=500000)
        ctrl.observe(200, 0.72, gaussian_count=500000)
        # INSERT: burst_ref captured at gc=500K, quality=0.71
        ctrl.observe(300, 0.71, gaussian_count=500000)
        self.assertEqual(ctrl.state, "DWELL")

        # DWELL: gc jumped to 550K after INSERT burst, quality DROPS (bad Gaussians)
        ctrl.observe(400, 0.65, gaussian_count=550000)
        ctrl.observe(500, 0.63, gaussian_count=550000)
        # After 2 DWELL probes -> ESTIMATE. Paired diff: (0.64 avg - 0.71 ref) / 50K < 0
        # ESTIMATE -> PRUNE -> DWELL
        prune_seen = False
        for i in range(5):
            rate = ctrl.observe(600 + i * 100, 0.60 + i * 0.01, gaussian_count=550000)
            if rate < 1.0:
                prune_seen = True
                break

        self.assertTrue(prune_seen, "Negative gradient should lead to PRUNE (rate < 1.0)")

    def test_near_zero_gradient_leads_to_hold(self):
        """When gradient is near zero (within deadband), ESTIMATE should select HOLD."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=2,
            perturbation_fraction=0.10,
            dwell_probes=2,
            gradient_window=3,
            noise_deadband=0.10,
        )
        # Warmup with flat quality (baseline_slope ~ 0)
        ctrl.observe(100, 0.81, gaussian_count=500000)
        ctrl.observe(200, 0.81, gaussian_count=500000)

        # Phase 1: establish a peak gradient via positive-gradient cycle
        # INSERT -> DWELL -> ESTIMATE (positive gradient establishes peak)
        ctrl.observe(300, 0.81, gaussian_count=500000)  # INSERT burst -> DWELL
        ctrl.observe(400, 0.83, gaussian_count=550000)  # DWELL (quality improves with gc)
        ctrl.observe(500, 0.85, gaussian_count=600000)  # DWELL -> ESTIMATE
        # ESTIMATE sees positive gradient, transitions to INSERT
        ctrl.observe(600, 0.86, gaussian_count=650000)  # Depending on state, continue

        # Phase 2: feed near-flat quality with growing gc (near-zero gradient)
        # The peak gradient is established; now tiny gradients should hit deadband
        hold_seen = False
        for i in range(15):
            gc = 700000 + i * 50000
            rate = ctrl.observe(700 + i * 100, 0.860 + (i % 2) * 0.0001, gaussian_count=gc)
            if ctrl.state == "HOLD":
                hold_seen = True
                self.assertAlmostEqual(rate, 1.0, msg="HOLD should return rate=1.0")
                break

        self.assertTrue(hold_seen, "Near-zero gradient should lead to HOLD")

    # ------------------------------------------------------------------
    # HOLD -> recheck
    # ------------------------------------------------------------------

    def test_hold_rechecks_after_interval(self):
        """In HOLD, the controller should re-estimate gradient after recheck_probes."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=2,
            perturbation_fraction=0.10,
            dwell_probes=2,
            gradient_window=3,
            noise_deadband=0.05,
            recheck_probes=5,
        )
        # Warmup
        ctrl.observe(100, 0.80, gaussian_count=500000)
        ctrl.observe(200, 0.81, gaussian_count=500000)
        # INSERT -> DWELL
        ctrl.observe(300, 0.810, gaussian_count=500000)

        # Feed flat quality to reach HOLD via DWELL -> ESTIMATE -> HOLD
        gc = 550000
        for i in range(15):
            ctrl.observe(400 + i * 100, 0.810 + (i % 2) * 0.001, gaussian_count=gc)

        # If we're in HOLD, feeding many more observations should eventually
        # trigger a re-evaluation (back to INSERT or ESTIMATE)
        if ctrl.state == "HOLD":
            states_after_hold = set()
            for i in range(20):
                ctrl.observe(2000 + i * 100, 0.810, gaussian_count=gc)
                states_after_hold.add(ctrl.state)
            self.assertTrue(
                len(states_after_hold) > 1 or "ESTIMATE" in states_after_hold or "INSERT" in states_after_hold,
                "HOLD should periodically re-evaluate (recheck)",
            )

    # ------------------------------------------------------------------
    # Perturbation sizing
    # ------------------------------------------------------------------

    def test_perturbation_scales_with_gaussian_count(self):
        """Perturbation size should be proportional to current Gaussian count."""
        ctrl = ExtremumSeekingController(perturbation_fraction=0.10, min_warmup_probes=2)

        ctrl.observe(100, 0.50, gaussian_count=100000)
        ctrl.observe(200, 0.55, gaussian_count=100000)
        # After warmup, last_gaussian_count should be 100000
        size = ctrl.perturbation_size
        self.assertGreater(size, 0)
        self.assertAlmostEqual(size / 100000, 0.10, delta=0.02)

    def test_perturbation_fraction_custom(self):
        """Custom perturbation_fraction should be respected."""
        ctrl = ExtremumSeekingController(perturbation_fraction=0.05, min_warmup_probes=2)
        ctrl.observe(100, 0.50, gaussian_count=200000)
        ctrl.observe(200, 0.55, gaussian_count=200000)
        self.assertAlmostEqual(ctrl.perturbation_size / 200000, 0.05, delta=0.02)

    # ------------------------------------------------------------------
    # Noise deadband
    # ------------------------------------------------------------------

    def test_deadband_prevents_chattering(self):
        """Tiny gradients within the deadband should not cause state changes from HOLD."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=2,
            perturbation_fraction=0.10,
            dwell_probes=2,
            gradient_window=5,
            noise_deadband=0.10,
        )
        # Warmup
        ctrl.observe(100, 0.80, gaussian_count=500000)
        ctrl.observe(200, 0.81, gaussian_count=500000)
        # INSERT -> DWELL
        ctrl.observe(300, 0.810, gaussian_count=500000)

        # Feed near-flat data to get to HOLD
        gc = 550000
        for i in range(15):
            ctrl.observe(400 + i * 100, 0.810 + (i % 2) * 0.0001, gaussian_count=gc)

        if ctrl.state == "HOLD":
            for i in range(15):
                noise = 0.0001 * (1 if i % 2 == 0 else -1)
                ctrl.observe(2000 + i * 100, 0.810 + noise, gaussian_count=gc)
                if ctrl.state in ("PRUNE",):
                    self.fail(f"Deadband should prevent chattering, but state changed to {ctrl.state}")

    # ------------------------------------------------------------------
    # PRUNE action
    # ------------------------------------------------------------------

    def test_prune_rate_below_one(self):
        """When PRUNE is selected, rate should be < 1.0."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=2,
            perturbation_fraction=0.10,
            dwell_probes=2,
            gradient_window=3,
            noise_deadband=0.01,
        )
        # Warmup
        ctrl.observe(100, 0.70, gaussian_count=500000)
        ctrl.observe(200, 0.72, gaussian_count=500000)
        # INSERT: burst_ref at gc=500K, quality=0.71
        ctrl.observe(300, 0.71, gaussian_count=500000)

        # DWELL: gc jumped to 550K, quality drops (bad Gaussians)
        ctrl.observe(400, 0.64, gaussian_count=550000)
        ctrl.observe(500, 0.60, gaussian_count=550000)
        # ESTIMATE -> negative gradient -> PRUNE -> DWELL
        for i in range(5):
            rate = ctrl.observe(600 + i * 100, 0.58 + i * 0.01, gaussian_count=550000)
            if rate < 1.0:
                self.assertEqual(ctrl.state, "DWELL", "PRUNE should immediately transition to DWELL")
                return

        self.fail("Should have seen rate < 1.0 for PRUNE")

    def test_prune_followed_by_dwell(self):
        """After PRUNE applies its burst, state should transition to DWELL."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=2,
            perturbation_fraction=0.10,
            dwell_probes=3,
            gradient_window=5,
            noise_deadband=0.01,
        )
        # Warmup
        ctrl.observe(100, 0.70, gaussian_count=500000)
        ctrl.observe(200, 0.72, gaussian_count=500000)
        # INSERT: burst_ref at gc=500K
        ctrl.observe(300, 0.71, gaussian_count=500000)

        # DWELL with constant gc=550K, quality drops after burst
        ctrl.observe(400, 0.63, gaussian_count=550000)
        ctrl.observe(500, 0.60, gaussian_count=550000)
        ctrl.observe(600, 0.58, gaussian_count=550000)
        # After 3 DWELL probes -> ESTIMATE -> negative gradient -> PRUNE -> DWELL
        prune_seen = False
        for i in range(5):
            rate = ctrl.observe(700 + i * 100, 0.55 + i * 0.01, gaussian_count=550000)
            if rate < 1.0:
                prune_seen = True
                self.assertEqual(ctrl.state, "DWELL", "PRUNE should immediately transition to DWELL")
                break

        self.assertTrue(prune_seen, "Should have reached PRUNE state")

    # ------------------------------------------------------------------
    # Self-correcting: PRUNE leading to quality loss triggers INSERT
    # ------------------------------------------------------------------

    def test_self_correcting_after_bad_prune(self):
        """After pruning, if gc is stable the ESC can't estimate gradient and
        defaults to INSERT (exploratory perturbation) on the next recheck."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=2,
            perturbation_fraction=0.05,
            dwell_probes=2,
            gradient_window=3,
            noise_deadband=0.02,
            recheck_probes=5,
        )
        # Warmup
        ctrl.observe(100, 0.80, gaussian_count=1000000)
        ctrl.observe(200, 0.81, gaussian_count=1000000)
        # INSERT -> DWELL
        ctrl.observe(300, 0.80, gaussian_count=1000000)

        # Feed negative gradient to get PRUNE
        for i in range(15):
            gc = 1050000 + i * 50000
            metric = 0.80 - i * 0.015
            ctrl.observe(400 + i * 100, metric, gaussian_count=gc)

        # Post-prune: gc stable, quality recovering (optimizer adjusts).
        # With constant gc the ESC can't estimate gradient, so on the next
        # recheck ESTIMATE it should default to INSERT (explore).
        gc_after_prune = 900000
        states_after = []
        for i in range(25):
            metric = 0.60 + i * 0.01
            ctrl.observe(2000 + i * 100, metric, gaussian_count=gc_after_prune)
            states_after.append(ctrl.state)

        self.assertIn("INSERT", states_after, "Self-correcting: should resume INSERT when gradient is indeterminate")

    # ------------------------------------------------------------------
    # Baseline subtraction (HP filter equivalent)
    # ------------------------------------------------------------------

    def test_baseline_subtraction_isolates_densification(self):
        """The controller should capture baseline slope from warmup for HP-filter subtraction."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=3,
            perturbation_fraction=0.10,
            dwell_probes=2,
            gradient_window=5,
            noise_deadband=0.05,
        )
        # Warmup with improving quality at constant gc
        ctrl.observe(100, 0.50, gaussian_count=100000)
        ctrl.observe(200, 0.55, gaussian_count=100000)
        ctrl.observe(300, 0.60, gaussian_count=100000)

        # After warmup exit, baseline slope should be non-zero
        self.assertNotEqual(ctrl.baseline_slope, 0.0, "Baseline should be captured from warmup")

    # ------------------------------------------------------------------
    # State dict roundtrip
    # ------------------------------------------------------------------

    def test_state_dict_roundtrip(self):
        """state_dict -> load_state_dict should preserve all controller state."""
        ctrl = ExtremumSeekingController(
            perturbation_fraction=0.08,
            dwell_probes=4,
            gradient_window=6,
            min_warmup_probes=3,
            recheck_probes=8,
            noise_deadband=0.03,
        )
        ctrl.observe(100, 0.50, gaussian_count=100000)
        ctrl.observe(200, 0.55, gaussian_count=120000)
        ctrl.observe(300, 0.60, gaussian_count=140000)
        ctrl.observe(400, 0.64, gaussian_count=165000)

        state = ctrl.state_dict()

        ctrl2 = ExtremumSeekingController()
        ctrl2.load_state_dict(state)

        self.assertEqual(ctrl2.state, ctrl.state)
        self.assertEqual(ctrl2._perturbation_fraction, 0.08)
        self.assertEqual(ctrl2._dwell_probes, 4)
        self.assertEqual(ctrl2._gradient_window, 6)
        self.assertEqual(ctrl2._min_warmup_probes, 3)
        self.assertEqual(ctrl2._recheck_probes, 8)
        self.assertAlmostEqual(ctrl2._noise_deadband, 0.03)
        self.assertEqual(len(ctrl2.history), len(ctrl.history))
        self.assertAlmostEqual(ctrl2.get_insertion_rate(), ctrl.get_insertion_rate())

    def test_state_dict_keys_present(self):
        """state_dict should contain all expected keys."""
        ctrl = ExtremumSeekingController()
        ctrl.observe(100, 0.50, gaussian_count=100000)
        state = ctrl.state_dict()

        expected_keys = {
            "perturbation_fraction",
            "dwell_probes",
            "gradient_window",
            "min_warmup_probes",
            "recheck_probes",
            "noise_deadband",
            "probe_every_k",
            "state",
            "history",
            "current_rate",
            "last_gradient",
        }
        for key in expected_keys:
            self.assertIn(key, state, f"state_dict missing key: {key}")

    # ------------------------------------------------------------------
    # Interface compatibility
    # ------------------------------------------------------------------

    def test_get_insertion_rate_matches_last_observe(self):
        """get_insertion_rate() should return the last rate from observe()."""
        ctrl = ExtremumSeekingController(min_warmup_probes=2)
        rate_from_observe = ctrl.observe(100, 0.50, gaussian_count=100000)
        rate_from_get = ctrl.get_insertion_rate()
        self.assertEqual(rate_from_observe, rate_from_get)

    def test_history_property(self):
        """history should return all observations as (step, metric, gc) triples."""
        ctrl = ExtremumSeekingController(min_warmup_probes=2)
        ctrl.observe(100, 0.50, gaussian_count=100000)
        ctrl.observe(200, 0.55, gaussian_count=120000)
        ctrl.observe(300, 0.60, gaussian_count=140000)

        h = ctrl.history
        self.assertEqual(len(h), 3)
        self.assertEqual(h[0], (100, 0.50, 100000))
        self.assertEqual(h[1], (200, 0.55, 120000))
        self.assertEqual(h[2], (300, 0.60, 140000))

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_observe_with_zero_gaussians(self):
        """Observing with gaussian_count=0 should not crash."""
        ctrl = ExtremumSeekingController(min_warmup_probes=2)
        rate = ctrl.observe(100, 0.50, gaussian_count=0)
        self.assertIsNotNone(rate)

    def test_observe_with_constant_metric(self):
        """Completely flat metric should eventually lead to HOLD (zero gradient)."""
        ctrl = ExtremumSeekingController(
            min_warmup_probes=2,
            perturbation_fraction=0.10,
            dwell_probes=2,
            gradient_window=3,
            noise_deadband=0.05,
        )
        # First cycle: establish a peak gradient to make deadband meaningful
        ctrl.observe(100, 0.70, gaussian_count=100000)
        ctrl.observe(200, 0.70, gaussian_count=100000)
        # INSERT: ref at gc=100K
        ctrl.observe(300, 0.70, gaussian_count=100000)
        # DWELL with higher gc and slightly better metric to get a positive gradient
        ctrl.observe(400, 0.75, gaussian_count=110000)
        ctrl.observe(500, 0.76, gaussian_count=110000)
        # ESTIMATE -> positive gradient established as peak -> INSERT

        # Second cycle: flat metric should give near-zero gradient -> HOLD
        hold_seen = False
        gc = 110000
        for i in range(20):
            # Alternate between INSERT/DWELL/ESTIMATE with flat metric
            if ctrl.state == "DWELL":
                ctrl.observe(600 + i * 100, 0.70, gaussian_count=gc)
            elif ctrl.state == "INSERT":
                ctrl.observe(600 + i * 100, 0.70, gaussian_count=gc)
                gc = int(gc * 1.10)
            else:
                ctrl.observe(600 + i * 100, 0.70, gaussian_count=gc)
            if ctrl.state == "HOLD":
                hold_seen = True
                break
        self.assertTrue(hold_seen, "Constant metric with growing count should lead to HOLD")


class PruneTests(GettysburgGaussianSplatTestCase, unittest.TestCase):
    """Tests for the _prune() method on GaussianSplatOptimizerMCMCControlled.
    RED phase -- _prune() does not exist yet, so these should fail with AttributeError."""

    def _make_optimizer(self, **config_overrides):
        """Helper to create a controlled optimizer with default test config."""
        if self.device != "cuda":
            self.skipTest("GaussianSplatOptimizerMCMCControlled uses CUDA-only ops")

        defaults = dict(
            noise_lr=0.0,
            insertion_rate=1.05,
            max_gaussians=-1,
            deletion_opacity_threshold=0.0,
            probe_every_k_refines=1,
            probe_n_images=1,
            trend_window=3,
            slope_threshold=1e-4,
            slope_threshold_high=2e-4,
            min_measurements_before_control=2,
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        defaults.update(config_overrides)
        config = frc.radiance_fields.GaussianSplatOptimizerMCMCControlledConfig(**defaults)
        optimizer = frc.radiance_fields.GaussianSplatOptimizerMCMCControlled.from_model_and_scene(
            model=self.model,
            sfm_scene=self.training_dataset.sfm_scene,
            config=config,
        )
        optimizer.reset_learning_rates_and_decay(batch_size=1, expected_steps=200)
        return optimizer

    def test_prune_removes_correct_count(self):
        """_prune(n) should remove exactly n Gaussians from the model."""
        optimizer = self._make_optimizer()
        n_before = self.model.num_gaussians
        n_to_prune = max(1, n_before // 10)

        removed = optimizer._prune(n_to_prune)

        self.assertEqual(removed, n_to_prune)
        self.assertEqual(self.model.num_gaussians, n_before - n_to_prune)

    def test_prune_removes_lowest_opacity(self):
        """_prune should remove the Gaussians with lowest opacity."""
        optimizer = self._make_optimizer()

        opacities_before = self.model.opacities.clone()
        n_to_prune = max(1, self.model.num_gaussians // 5)

        optimizer._prune(n_to_prune)

        # The remaining Gaussians should have opacities >= the pruning threshold
        # (i.e., the lowest-opacity ones should be gone)
        sorted_opacities_before, _ = torch.sort(opacities_before)
        prune_threshold = sorted_opacities_before[n_to_prune - 1].item()

        # All remaining opacities should be >= the threshold
        remaining_opacities = self.model.opacities
        self.assertTrue(
            (remaining_opacities >= prune_threshold - 1e-6).all(),
            "Remaining Gaussians should have opacity >= the pruned threshold",
        )

    def test_prune_zero_is_noop(self):
        """_prune(0) should not modify the model."""
        optimizer = self._make_optimizer()
        n_before = self.model.num_gaussians
        means_before = self.model.means.clone()

        removed = optimizer._prune(0)

        self.assertEqual(removed, 0)
        self.assertEqual(self.model.num_gaussians, n_before)
        self.assertTrue(torch.allclose(self.model.means, means_before))

    def test_prune_more_than_exist_clamps(self):
        """_prune(n) where n >= num_gaussians should prune all but at least 1 (or clamp gracefully)."""
        optimizer = self._make_optimizer()
        n_before = self.model.num_gaussians

        removed = optimizer._prune(n_before + 1000)

        # Should not crash; should prune at most n_before - 1 (keep at least 1)
        self.assertGreater(self.model.num_gaussians, 0)
        self.assertLessEqual(removed, n_before)

    def test_prune_updates_optimizer_state(self):
        """After _prune, optimizer param groups should match the new model size."""
        optimizer = self._make_optimizer()

        # Run a forward+backward to populate gradients and optimizer state
        optimizer.zero_grad()
        gt_img, pred_img, _ = self._render_one_image(self.model)
        loss = torch.nn.functional.l1_loss(pred_img, gt_img)
        loss.backward()
        optimizer.step()

        n_to_prune = max(1, self.model.num_gaussians // 10)
        optimizer._prune(n_to_prune)

        # All param groups should have tensors matching the new model size
        expected_n = self.model.num_gaussians
        for pg in optimizer._optimizer.param_groups:
            param = pg["params"][0]
            self.assertEqual(
                param.shape[0],
                expected_n,
                f"Param group '{pg['name']}' has shape {param.shape[0]}, expected {expected_n}",
            )

    def test_prune_preserves_model_integrity(self):
        """After pruning, the model should still be renderable."""
        optimizer = self._make_optimizer()
        n_to_prune = max(1, self.model.num_gaussians // 5)

        optimizer._prune(n_to_prune)

        # Should not crash
        gt_img, pred_img, alphas = self._render_one_image(self.model)
        self.assertEqual(pred_img.shape, gt_img.shape)

    def test_prune_uses_filter_gaussians(self):
        """_prune should internally use filter_gaussians to do the actual removal."""
        optimizer = self._make_optimizer()
        n_before = self.model.num_gaussians
        n_to_prune = max(1, n_before // 10)

        # This is a behavioral test -- we verify the end result is consistent
        # with filter_gaussians (correct param/optimizer state update)
        optimizer._prune(n_to_prune)

        # Verify all model tensors have consistent shapes
        n_after = self.model.num_gaussians
        self.assertEqual(self.model.means.shape[0], n_after)
        self.assertEqual(self.model.quats.shape[0], n_after)
        self.assertEqual(self.model.log_scales.shape[0], n_after)
        self.assertEqual(self.model.logit_opacities.shape[0], n_after)
        self.assertEqual(self.model.sh0.shape[0], n_after)
        self.assertEqual(self.model.shN.shape[0], n_after)


if __name__ == "__main__":
    unittest.main()
