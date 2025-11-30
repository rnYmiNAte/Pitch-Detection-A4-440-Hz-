#!/usr/bin/env python3
"""Detect occurrences of A4 ~= 440 Hz in MP3/WAV files.
Output CSV with columns: file, start_sec, end_sec, peak_freq_hz, peak_db

Simple approach:
- load audio with librosa (mono, sr=22050)
- use short-time Fourier transform (STFT)
- compute spectral centroid and/or pick strongest frequency from each frame
- mark frames where pitch ~= 440Hz within tolerance (cents)
- group adjacent frames into events and record start/end times

This is not a perfect musical pitch tracker but works reliably for clear tones.
"""

import argparse
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import math
from tqdm import tqdm

A4 = 440.0


def hz_to_cents(hz, ref=440.0):
    return 1200.0 * math.log2(hz / ref)


def cents_to_hz(cents, ref=440.0):
    return ref * (2.0 ** (cents / 1200.0))


def detect_a4_in_file(path, sr=22050, hop_length=512, n_fft=2048, tol_cents=50, min_frames=3):
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return []

    # magnitude spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # find index of nearest frequency bin to each frequency
    # compute the dominant frequency per frame
    dominant_idx = np.argmax(S, axis=0)
    dominant_freqs = freqs[dominant_idx]
    dominant_db = librosa.amplitude_to_db(S[dominant_idx, np.arange(S.shape[1])], ref=np.max)

    # convert difference in cents
    cents_diff = [hz_to_cents(f, A4) for f in dominant_freqs]
    mask = np.isfinite(dominant_freqs) & (np.abs(cents_diff) <= tol_cents)

    # group consecutive True frames
    events = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            end = i - 1
            if (end - start + 1) >= min_frames:
                # compute time bounds
                start_sec = start * hop_length / sr
                end_sec = (end * hop_length + n_fft) / sr
                # pick peak frequency and db within event
                idxs = np.arange(start, end + 1)
                peak_idx = idxs[np.argmax(dominant_db[idxs])]
                events.append({
                    'start_sec': float(start_sec),
                    'end_sec': float(end_sec),
                    'peak_freq_hz': float(dominant_freqs[peak_idx]),
                    'peak_db': float(dominant_db[peak_idx])
                })
            start = None
    # handle tail
    if start is not None:
        end = len(mask) - 1
        if (end - start + 1) >= min_frames:
            start_sec = start * hop_length / sr
            end_sec = (end * hop_length + n_fft) / sr
            idxs = np.arange(start, end + 1)
            peak_idx = idxs[np.argmax(dominant_db[idxs])]
            events.append({
                'start_sec': float(start_sec),
                'end_sec': float(end_sec),
                'peak_freq_hz': float(dominant_freqs[peak_idx]),
                'peak_db': float(dominant_db[peak_idx])
            })

    return events


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glob', default='**/*', help='glob or path to search for audio files')
    parser.add_argument('--tolerance-cents', type=float, default=50.0, help='tolerance in cents around A4')
    parser.add_argument('--min-duration', type=float, default=0.05, help='minimum event duration in seconds')
    parser.add_argument('--out', default='results/pitch_detection_results.csv')
    args = parser.parse_args()

    Path('results').mkdir(parents=True, exist_ok=True)

    files = []
    for p in glob.glob(args.glob, recursive=True):
        if p.lower().endswith(('.mp3', '.wav')) and os.path.isfile(p):
            files.append(p)

    records = []
    for f in tqdm(files, desc='Files'):
        try:
            events = detect_a4_in_file(f, tol_cents=args.tolerance_cents)
            # filter by min_duration
            for ev in events:
                dur = ev['end_sec'] - ev['start_sec']
                if dur >= args.min_duration:
                    records.append({
                        'file': f,
                        'start_sec': ev['start_sec'],
                        'end_sec': ev['end_sec'],
                        'peak_freq_hz': ev['peak_freq_hz'],
                        'peak_db': ev['peak_db']
                    })
        except Exception as e:
            print(f'ERROR processing {f}: {e}')

    if len(records) == 0:
        print('No A4 events detected.')

    df = pd.DataFrame(records, columns=['file', 'start_sec', 'end_sec', 'peak_freq_hz', 'peak_db'])
    df.to_csv(args.out, index=False)
    print(f'Wrote results to {args.out}')
