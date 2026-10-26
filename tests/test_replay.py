"""Replay regression tests; run with the project's Python inside Docker."""

import unittest

import torch

from supermimic.buffer import ReplayBuffer


class ReplayTests(unittest.TestCase):
    def make_buffer(self):
        buffer = ReplayBuffer(5, 2, "cpu")
        for step in range(7):
            transition = {key: torch.full_like(value[0], step) for key, value in buffer.data.items()}
            transition["done"][:] = step % 3 == 2
            transition["ep_step"][:] = step % 3
            buffer.add(transition)
        return buffer

    def test_windows_after_wrap_and_episode_end(self):
        torch.manual_seed(0)
        buffer = self.make_buffer()
        sample = buffer.sample_windows(256, 3)
        starts = sample["s"][:, 0, 0].long()
        self.assertEqual(set(starts.tolist()), {2, 3, 4})
        for start, states, mask in zip(starts.tolist(), sample["s"], sample["mask"]):
            self.assertEqual(states[:, 0].tolist(), [start, start + 1, start + 2])
            self.assertEqual(mask.tolist(), {2: [1, 0, 0], 3: [1, 1, 1], 4: [1, 1, 0]}[start])

    def test_history_stays_within_episode_and_retained_buffer(self):
        torch.manual_seed(1)
        buffer = self.make_buffer()
        starts, history = buffer.sample_starts(256, 4)
        expected = {2: [2, 2, 2, 2], 3: [3, 3, 3, 3], 4: [3, 3, 3, 4],
                    5: [3, 3, 4, 5], 6: [6, 6, 6, 6]}
        self.assertEqual(set(starts["f"].tolist()), set(expected))
        for frame, frames in zip(starts["f"].tolist(), history["f"].tolist()):
            self.assertEqual(frames, expected[frame])


if __name__ == "__main__":
    unittest.main()
