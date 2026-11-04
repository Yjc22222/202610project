"""Deterministic windows, including complete coverage of long recordings."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Segment:
    start_sample: int
    end_sample: int
    samples: np.ndarray


def make_segments(audio, window_samples=64600, hop_samples=32300, mode='sliding'):
    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError('Expected non-empty mono audio.')
    if window_samples <= 0 or not 0 < hop_samples <= window_samples:
        raise ValueError('Require 0 < hop_samples <= window_samples.')
    if mode not in ('sliding', 'first'):
        raise ValueError('Unknown segment mode: ' + mode)
    size = len(audio)
    if size <= window_samples:
        padded = np.tile(audio, (window_samples + size - 1) // size)[:window_samples]
        return [Segment(0, size, np.ascontiguousarray(padded, dtype=np.float32))]
    starts = [0] if mode == 'first' else list(range(0, size - window_samples + 1, hop_samples))
    if mode == 'sliding' and starts[-1] != size - window_samples:
        starts.append(size - window_samples)
    return [Segment(start, start + window_samples, audio[start:start + window_samples]) for start in starts]


def aggregate_scores(scores, method='mean'):
    values = np.asarray(scores, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Expected finite segment scores.')
    if method == 'mean':
        return float(np.mean(values))
    if method == 'median':
        return float(np.median(values))
    if method == 'q10':
        return float(np.quantile(values, .1))
    raise ValueError('Unknown aggregation method: ' + method)
