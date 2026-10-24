import io

import numpy as np

from earshot_arrays.meter import map_channels, tap_channel

FS = 1000


def test_tap_channel_picks_the_tapped_mic():
    rng = np.random.default_rng(0)
    block = 0.001 * rng.standard_normal((16, 100))
    noise = np.full(16, 0.001)
    assert tap_channel(block, noise) is None  # just noise
    block[7, 40] = 0.5
    block[6, 41] = 0.05  # neighbour hears it too, much quieter
    assert tap_channel(block, noise) == 7
    block[6, 41] = 0.4  # two equally loud channels: ambiguous
    assert tap_channel(block, noise) is None


class FakeTaps:
    """0.1 s blocks of noise; after each prompt (drain) the next channel of `order` is tapped."""

    def __init__(self, order):
        self.fs, self.order, self.i, self.n = FS, list(order), 0, len(order)
        self.rng = np.random.default_rng(1)
        self.tap_in = None

    def read(self):
        self.i += 1
        b = 0.001 * self.rng.standard_normal((self.n, 100))
        if self.tap_in is not None:
            self.tap_in -= 1
            if self.tap_in == 0:
                b[self.order.pop(0), 30] = 0.5
                self.tap_in = None
        return b

    def drain(self):
        self.tap_in = 3


def test_map_channels_returns_tap_order():
    order = [3, 0, 2, 1]
    assert map_channels(FakeTaps(order), 4, out=io.StringIO()) == [3, 0, 2, 1]
