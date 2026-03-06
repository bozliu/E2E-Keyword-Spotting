from __future__ import annotations

import numpy as np

from kws.demo.realtime import AudioRingBuffer


def test_audio_ring_buffer_not_ready_until_capacity() -> None:
    buf = AudioRingBuffer(capacity=8)
    buf.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert buf.is_ready is False
    assert buf.size == 3
    assert np.allclose(buf.latest(), np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_audio_ring_buffer_keeps_latest_samples_on_wrap() -> None:
    buf = AudioRingBuffer(capacity=5)
    buf.append(np.array([1, 2, 3], dtype=np.float32))
    buf.append(np.array([4, 5, 6], dtype=np.float32))

    assert buf.is_ready is True
    assert buf.size == 5
    assert np.allclose(buf.latest(), np.array([2, 3, 4, 5, 6], dtype=np.float32))


def test_audio_ring_buffer_large_chunk_replaces_full_window() -> None:
    buf = AudioRingBuffer(capacity=4)
    buf.append(np.array([0.0, 1.0], dtype=np.float32))
    buf.append(np.array([7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32))

    assert buf.is_ready is True
    assert buf.size == 4
    assert np.allclose(buf.latest(), np.array([8.0, 9.0, 10.0, 11.0], dtype=np.float32))
