"""Deterministic interaction regressions; no webcam or GUI is opened."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from main import CFG, HandData, HandTracker, Object2D, ObjectManipulator, fit_camera_frame


def hand(x=150, gap=0.1, pinching=True):
    return HandData((x + 5, 150), (x - 5, 150), (0, 0), pinching, gap, 0)


class InteractionTests(unittest.TestCase):
    def app(self):
        app = ObjectManipulator.__new__(ObjectManipulator)
        app.objects = [Object2D(100, 100, 100, 100, (0, 255, 0), 'test')]
        app._clear_grab()
        app.offset = (0, 0)
        return app

    def test_either_hand_owns_grab_and_other_resizes(self):
        for owner in ('Right', 'Left'):
            with self.subTest(owner=owner):
                app = self.app()
                hands = {owner: hand(), 'Left' if owner == 'Right' else 'Right': hand(400)}
                app.handle_interaction(hands['Right'], hands['Left'])
                self.assertEqual(app.grab_hand, owner)
                other = hands['Left' if owner == 'Right' else 'Right']
                other.pinch_distance = 0.15
                app.handle_interaction(hands['Right'], hands['Left'])
                self.assertGreater(app.objects[0].w, 100)
                position = app.objects[0].x
                other.pinch_distance = 0.1 * app.smooth_scale
                app.handle_interaction(hands['Right'], hands['Left'])
                self.assertAlmostEqual(app.objects[0].x, position)

    def test_release_debounce_and_tracking_loss(self):
        app = self.app()
        app.handle_interaction(hand(), None)
        for _ in range(CFG.RELEASE_FRAMES - 1):
            app.handle_interaction(hand(pinching=False), None)
            self.assertIsNotNone(app.grabbed_idx)
        app.handle_interaction(hand(pinching=False), None)
        self.assertIsNone(app.grabbed_idx)
        app.handle_interaction(hand(), None)
        for _ in range(CFG.LOST_TOLERANCE + 1):
            app.handle_interaction(None, hand(400))
        self.assertIsNone(app.grabbed_idx)

    def test_letterbox_preserves_frame(self):
        source = np.full((100, 200, 3), 255, dtype=np.uint8)
        canvas = fit_camera_frame(source, 960, 600)
        self.assertEqual(canvas.shape, (600, 960, 3))
        self.assertTrue(np.all(canvas[60:540] == 255))
        self.assertTrue(np.all(canvas[:60] != 255))


class TrackerTests(unittest.TestCase):
    def test_smoothing_history_midpoint_velocity_and_missing_hand(self):
        tracker = HandTracker.__new__(HandTracker)
        tracker.smooth_state = {}
        tracker.last_timestamp_ms = -1
        tracker.hands = Mock()
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        landmarks[5].x, landmarks[17].x = 0.4, 0.6
        landmarks[9].y = 0.3
        result = SimpleNamespace(hand_landmarks=[landmarks], handedness=[[
            SimpleNamespace(category_name='Left')]])
        tracker.hands.detect_for_video.return_value = result
        frame = np.zeros((600, 960, 3), dtype=np.uint8)
        first, _ = tracker.process(frame)
        self.assertTrue(first.is_pinching)
        moved = copy.deepcopy(landmarks)
        for point in moved:
            point.x += 0.02
        result.hand_landmarks = [moved]
        second, _ = tracker.process(frame)
        self.assertAlmostEqual(tracker.smooth_state['Right']['prev'][0], moved[8].x * 960)
        self.assertAlmostEqual(second.velocity[0], second.pinch_point[0] - first.pinch_point[0])
        # A fresh hand with a 0.35-palm gap must stay open; the old
        # 0.48 threshold classified this clearly separated pose as a pinch.
        tracker.smooth_state.clear()
        separated = copy.deepcopy(landmarks)
        palm_size = 0.7 * (0.2 * 960) + 0.3 * (0.2 * 600)
        separated[8].x += 0.35 * palm_size / 960
        result.hand_landmarks = [separated]
        open_hand, _ = tracker.process(frame)
        self.assertFalse(open_hand.is_pinching)
        self.assertEqual(tracker.hands.detect_for_video.call_args.args[0].width, CFG.TRACKING_WIDTH)
        # Keep the other hand visible while the original disappears.
        result.handedness = [[SimpleNamespace(category_name='Right')]]
        tracker.process(frame)
        self.assertNotIn('Right', tracker.smooth_state)
        timestamps = [call.args[1] for call in tracker.hands.detect_for_video.call_args_list]
        self.assertTrue(all(a < b for a, b in zip(timestamps, timestamps[1:])))


if __name__ == '__main__':
    unittest.main()
