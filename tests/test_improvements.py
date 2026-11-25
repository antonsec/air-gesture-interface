"""Time, geometry, ownership recovery and calibration contracts without a camera."""
import copy
import math
import unittest
from unittest.mock import patch

from main import (CFG, Object2D, PinchCalibration, collision_overlap,
                  constrain_to_canvas, update_physics, assign_hand_labels)
from test_interaction import hand
import test_interaction


def obj(x=200, y=200, w=100, h=50, angle=0):
    return Object2D(x, y, w, h, (0, 255, 0), 'test', angle=angle)


def anchor_position(app):
    item = app.objects[app.grabbed_idx]
    u, v = item.axes()
    cx, cy = item.center
    ax, ay = app.anchor[0] * item.w, app.anchor[1] * item.h
    return cx + ax * u[0] + ay * v[0], cy + ax * u[1] + ay * v[1]


class PhysicsTests(unittest.TestCase):
    def test_one_second_of_motion_is_equal_at_four_frame_rates(self):
        results = []
        for fps in (15, 30, 60, 120):
            item = obj()
            item.vx, item.vy, item.va = 150, 40, 90
            for _ in range(fps):
                update_physics([item], 960, 600, False, dt=1 / fps)
            results.append((item.x, item.y, item.angle, item.vx, item.vy, item.va))
        for result in results[1:]:
            for actual, expected in zip(result, results[0]):
                self.assertAlmostEqual(actual, expected, places=7)

    def test_no_friction_keeps_constant_velocity(self):
        item = obj()
        item.vx = 100
        with patch.object(CFG, 'FRICTION', 1.0):
            update_physics([item], 960, 600, False, dt=0.1)
        self.assertAlmostEqual(item.x, 210)
        self.assertEqual(item.vx, 100)

    def test_stall_is_bounded_and_pause_preserves_state(self):
        item = obj()
        item.vx = 300
        expected = copy.deepcopy(item)
        update_physics([item], 960, 600, False, dt=10)
        update_physics([expected], 960, 600, False, dt=CFG.MAX_PHYSICS_DT)
        self.assertEqual(item, expected)
        unchanged = copy.deepcopy(item)
        update_physics([item], 960, 600, True, dt=1)
        self.assertEqual(item, unchanged)

    def test_rotated_corners_stay_inside_walls_and_bounce_outward_only(self):
        for x, y, vx, vy in ((-20, -20, -80, -80), (900, 580, 80, 80)):
            item = obj(x=x, y=y, angle=45)
            item.vx, item.vy = vx, vy
            constrain_to_canvas(item, 960, 600)
            for px, py in item.corners():
                self.assertTrue(-1e-7 <= px <= 960 + 1e-7)
                self.assertTrue(-1e-7 <= py <= 600 + 1e-7)
            self.assertLess(item.vx * vx, 0)
            self.assertLess(item.vy * vy, 0)

    def test_sat_rejects_separated_rotated_shapes_with_overlapping_bounds(self):
        a = obj(x=200, y=200, w=200, h=10, angle=45)
        b = copy.deepcopy(a)
        b.y += 30
        self.assertIsNone(collision_overlap(a, b))

    def test_fully_overlapping_objects_separate(self):
        a, b = obj(), obj()
        update_physics([a, b], 960, 600, False)
        self.assertIsNone(collision_overlap(a, b))

    def test_held_object_is_fixed_while_collision_moves_free_object(self):
        held, free = obj(), obj(x=250, angle=30)
        original = copy.deepcopy(held)
        update_physics([held, free], 960, 600, False, held_idx=0)
        self.assertEqual(held, original)
        self.assertIsNone(collision_overlap(held, free))


class RecoveryAndAnchorTests(unittest.TestCase):
    def app(self):
        return test_interaction.InteractionTests().app()

    def test_rotation_and_scale_keep_off_center_grab_anchor(self):
        app = self.app()
        owner, other = hand(180), hand(400)
        app.handle_interaction(owner, other)
        owner.angle = 70
        other.pinch_distance = 0.15
        with patch.object(CFG, 'HOLD_FOLLOW', 1.0):
            app.handle_interaction(owner, other)
        for actual, expected in zip(anchor_position(app), owner.pinch_point):
            self.assertAlmostEqual(actual, expected)
        self.assertGreater(app.objects[0].w, 100)
        self.assertGreater(app.objects[0].angle, 60)

    def test_recovery_reanchors_without_teleport_or_throw(self):
        app = self.app()
        app.handle_interaction(hand(), None)
        before = copy.deepcopy(app.objects[0])
        app.handle_interaction(None, hand(400), dt=0.1)
        returned = hand(210)
        returned.velocity = (1000, 1000)
        returned.angle = 150
        app.handle_interaction(returned, hand(400))
        self.assertEqual(app.objects[0], before)
        self.assertEqual(app.grab_hand, 'Right')
        self.assertEqual(app.angular_velocity, 0)

    def test_release_of_recovered_open_hand_has_no_throw(self):
        app = self.app()
        app.handle_interaction(hand(), None)
        app.handle_interaction(None, None, dt=0.1)
        app.handle_interaction(hand(200, pinching=False), None)
        self.assertIsNone(app.grabbed_idx)
        self.assertEqual(app.objects[0].vx, 0)

    def test_loss_timeout_and_release_use_seconds_at_different_fps(self):
        for fps in (15, 30, 60):
            app = self.app()
            app.handle_interaction(hand(), None)
            for _ in range(math.ceil(CFG.LOST_SECONDS * fps) + 1):
                app.handle_interaction(None, None, dt=1 / fps)
            self.assertIsNone(app.grabbed_idx)
            app.handle_interaction(hand(), None)
            for _ in range(math.ceil(CFG.RELEASE_SECONDS * fps)):
                app.handle_interaction(hand(pinching=False), None, dt=1 / fps)
            self.assertIsNone(app.grabbed_idx)

    def test_hand_association_survives_label_flip_and_crossing(self):
        states = {
            'Right': {'wrist': (460, 200), 'wrist_velocity': (1200, 0), 'last_seen': 1},
            'Left': {'wrist': (500, 200), 'wrist_velocity': (-1200, 0), 'last_seen': 1},
        }
        detections = [{'wrist': (500, 200), 'label': 'Left'},
                      {'wrist': (460, 200), 'label': 'Right'}]
        self.assertEqual(assign_hand_labels(detections, states, 1 + 1 / 30), ('Right', 'Left'))


class CalibrationTests(unittest.TestCase):
    def sample(self, calibration, ratio, label='Right'):
        sample = hand()
        sample.gap_ratio = ratio
        for _ in range(35):
            calibration.update({label: sample}, 1 / 30)

    def test_two_poses_calibrate_one_hand_and_reset(self):
        calibration = PinchCalibration()
        calibration.advance({'Right': hand(), 'Left': hand()})
        self.assertFalse(calibration.active)
        calibration.advance({'Right': hand()})
        self.sample(calibration, 0.9)
        self.assertEqual(calibration.phase, 'ready_pinch')
        calibration.advance({'Right': hand()})
        self.sample(calibration, 0.12)
        close, release = calibration.thresholds['Right']
        self.assertTrue(0.12 < close < release < 0.9)
        self.assertNotIn('Left', calibration.thresholds)
        self.assertFalse(calibration.active)
        calibration.reset()
        self.assertEqual(calibration.thresholds, {})

    def test_missing_hand_times_out_without_overwriting_existing_thresholds(self):
        calibration = PinchCalibration()
        calibration.thresholds['Right'] = (0.2, 0.35)
        calibration.advance({'Right': hand()})
        calibration.update({}, 10.1)
        self.assertFalse(calibration.active)
        self.assertEqual(calibration.thresholds['Right'], (0.2, 0.35))

    def test_similar_poses_are_rejected(self):
        calibration = PinchCalibration()
        calibration.advance({'Right': hand()})
        self.sample(calibration, 0.6)
        calibration.advance({'Right': hand()})
        self.sample(calibration, 0.5)
        self.assertEqual(calibration.thresholds, {})
        self.assertIn('retry', calibration.message)

class TrackerIntegrationTests(unittest.TestCase):
    def tracker(self):
        from main import HandTracker
        from unittest.mock import Mock
        tracker = HandTracker.__new__(HandTracker)
        tracker.smooth_state = {}
        tracker.last_timestamp_ms = -1
        tracker.calibration = PinchCalibration()
        tracker.hands = Mock()
        return tracker

    def result(self, ratio=0.0, label='Left'):
        from types import SimpleNamespace
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        points[5].x, points[17].x, points[9].y = 0.4, 0.6, 0.3
        palm = 0.7 * 0.2 * 960 + 0.3 * 0.2 * 600
        points[8].x += ratio * palm / 960
        return SimpleNamespace(hand_landmarks=[points], handedness=[[
            SimpleNamespace(category_name=label)]])

    def test_per_hand_thresholds_are_used_in_normal_tracking_pipeline(self):
        import numpy as np
        tracker = self.tracker()
        tracker.calibration.thresholds['Right'] = (0.38, 0.5)
        tracker.hands.detect_for_video.return_value = self.result(0.35)
        first, _ = tracker.process(np.zeros((600, 960, 3), dtype=np.uint8))
        self.assertTrue(first.is_pinching)
        tracker.smooth_state.clear()
        tracker.calibration.reset()
        second, _ = tracker.process(np.zeros((600, 960, 3), dtype=np.uint8))
        self.assertFalse(second.is_pinching)

    def test_elapsed_gap_without_intermediate_frames_is_marked_reacquired(self):
        import numpy as np
        tracker = self.tracker()
        tracker.hands.detect_for_video.return_value = self.result()
        with patch('main.time.monotonic_ns', side_effect=[1_000_000_000, 2_000_000_000]):
            frame = np.zeros((600, 960, 3), dtype=np.uint8)
            tracker.process(frame)
            returned, _ = tracker.process(frame)
        self.assertTrue(returned.reacquired)
        self.assertEqual(returned.velocity, (0.0, 0.0))
