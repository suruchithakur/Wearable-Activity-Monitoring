import numpy as np
from scipy.signal import find_peaks, butter, filtfilt


def _bandpass(signal, fs, low=0.5, high=5.0, order=4):
    """Apply bandpass filter with safety checks."""
    if len(signal) < 15 or fs <= 0:
        return signal - np.mean(signal)
    try:
        nyq = 0.5 * fs
        lo = max(low / nyq, 0.001)
        hi = min(high / nyq, 0.999)
        if lo >= hi:
            return signal - np.mean(signal)
        b, a = butter(order, [lo, hi], btype='band')
        return filtfilt(b, a, signal)
    except Exception:
        return signal - np.mean(signal)


def _count_steps_peak_detection(signal, fs, height_threshold=None, min_distance_sec=0.3):
    """Count steps using adaptive peak detection on filtered acceleration magnitude."""
    if len(signal) < 10 or fs <= 0:
        return 0

    filtered = _bandpass(signal, fs, low=0.5, high=5.0)

    # Adaptive height threshold based on signal statistics
    if height_threshold is None:
        sig_std = np.std(filtered)
        sig_mad = np.median(np.abs(filtered - np.median(filtered)))
        # Use a fraction of the signal's variability as threshold
        height_threshold = max(0.05, min(sig_mad * 0.5, sig_std * 0.3))

    min_distance = max(int(min_distance_sec * fs), 1)

    peaks, properties = find_peaks(filtered, height=height_threshold, distance=min_distance)

    return len(peaks)


def _count_steps_autocorrelation(signal, fs):
    """Count steps using autocorrelation-based cadence detection."""
    if len(signal) < 200 or fs <= 0:
        return 0

    filtered = _bandpass(signal, fs, low=0.5, high=5.0)

    # Normalize
    filtered = filtered - np.mean(filtered)
    if np.std(filtered) < 1e-10:
        return 0
    filtered = filtered / np.std(filtered)

    # Compute autocorrelation for step-period lags (0.3s to 1.5s)
    min_lag = int(0.3 * fs)   # ~3.3 Hz (fast running)
    max_lag = min(int(1.5 * fs), len(filtered) // 2)  # ~0.67 Hz (slow walking)

    if min_lag >= max_lag:
        return 0

    # Efficient autocorrelation via slicing
    n = len(filtered)
    autocorr = np.array([np.sum(filtered[:n - lag] * filtered[lag:]) / (n - lag)
                         for lag in range(min_lag, max_lag)])

    if len(autocorr) == 0:
        return 0

    # Find dominant peak in autocorrelation
    peaks, props = find_peaks(autocorr, height=0.1)
    if len(peaks) == 0:
        return 0

    # Best peak = highest autocorrelation
    best_idx = peaks[np.argmax(props['peak_heights'])]
    step_period = (best_idx + min_lag) / fs  # in seconds
    autocorr_strength = props['peak_heights'][np.argmax(props['peak_heights'])]

    if step_period <= 0 or autocorr_strength < 0.1:
        return 0

    # Total duration
    duration = len(signal) / fs

    # Estimate steps from cadence
    steps = int(duration / step_period)

    return steps


def _segment_activity(mag, fs, window_sec=3.0):
    """Segment signal into active (walking/running) and inactive (standing/cycling) periods.
    Returns a boolean mask at the original signal's sample rate."""
    window_size = int(window_sec * fs)
    if window_size < 10 or len(mag) < window_size:
        return np.ones(len(mag), dtype=bool)

    n_windows = len(mag) // window_size
    is_active = np.zeros(len(mag), dtype=bool)

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        w = mag[start:end]

        # Compute energy in walking frequency band
        w_filtered = _bandpass(w, fs, low=0.5, high=5.0)
        energy = np.std(w_filtered)

        # Active if energy is above threshold (standing/cycling have low walk-band energy)
        if energy > 0.15:
            is_active[start:end] = True

    # Handle remaining samples
    if n_windows * window_size < len(mag):
        is_active[n_windows * window_size:] = is_active[n_windows * window_size - 1] if n_windows > 0 else True

    return is_active


def count_steps(recording, watch_loc=None):
    """Count steps using combined peak detection + autocorrelation approach.

    Args:
        recording: Recording object with sensor data
        watch_loc: Optional watch location (0=wrist, 1=belt, 2=ankle) for tuning
    """
    if 'ax' not in recording.data:
        return 0

    ax = recording.data['ax'].values
    ay = recording.data['ay'].values
    az = recording.data['az'].values
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    fs = recording.data['ax'].samplerate

    if len(mag) < 20 or fs <= 0:
        return 0

    # Segment into active/inactive periods
    active_mask = _segment_activity(mag, fs)
    active_signal = mag[active_mask]

    if len(active_signal) < 20:
        return 0

    # Adjust parameters based on watch location
    if watch_loc == 2:  # Ankle - strongest step signal
        height_factor = 0.2
        min_dist_sec = 0.25
    elif watch_loc == 1:  # Belt - moderate signal
        height_factor = 0.15
        min_dist_sec = 0.3
    elif watch_loc == 0:  # Wrist - weakest, most variable
        height_factor = 0.1
        min_dist_sec = 0.3
    else:
        height_factor = 0.15
        min_dist_sec = 0.3

    # Method 1: Peak detection on active segments
    filtered = _bandpass(active_signal, fs, low=0.5, high=5.0)
    sig_std = np.std(filtered)
    height_thresh = max(0.05, sig_std * height_factor)
    min_dist = max(int(min_dist_sec * fs), 1)

    peaks, _ = find_peaks(filtered, height=height_thresh, distance=min_dist)
    steps_peak = len(peaks)

    # Method 2: Autocorrelation on active segments
    steps_autocorr = _count_steps_autocorrelation(active_signal, fs)

    # Combine: use weighted average, favoring peak detection but using autocorrelation as sanity check
    if steps_autocorr > 0 and steps_peak > 0:
        ratio = steps_peak / steps_autocorr
        if 0.7 < ratio < 1.4:
            # Methods agree, use peak detection (more precise)
            steps = steps_peak
        elif ratio <= 0.7:
            # Peak detection underestimates, use average
            steps = int(0.6 * steps_autocorr + 0.4 * steps_peak)
        else:
            # Peak detection overestimates, use average
            steps = int(0.4 * steps_autocorr + 0.6 * steps_peak)
    elif steps_peak > 0:
        steps = steps_peak
    elif steps_autocorr > 0:
        steps = steps_autocorr
    else:
        steps = 0

    return max(0, steps)


if __name__ == "__main__":
    import os
    import pandas as pd
    from mhealth_activity import Recording

    metadata = pd.read_csv('train_metadata.csv')
    # Filter rows where step_count is valid (>0)
    valid_steps = metadata[metadata['step_count'] > 0]

    print(f"Testing Step counter on {len(valid_steps)} recordings with valid step counts...")
    errors = []
    abs_errors = []

    for i, row in valid_steps.iterrows():
        f_path = os.path.join('data/train', row['filename'])
        rec = Recording(f_path)
        watch_loc = int(row['watch_loc'])
        pred = count_steps(rec, watch_loc=watch_loc)
        actual = int(row['step_count'])
        if actual > 0:
            error = abs(pred - actual) / actual
            abs_err = abs(pred - actual)
            print(f"File: {row['filename']}, WatchLoc: {watch_loc}, Pred: {pred}, Actual: {actual}, "
                  f"RelErr: {error:.2%}, AbsErr: {abs_err}")
            errors.append(error)
            abs_errors.append(abs_err)

    print(f"\nMean Relative Error: {np.mean(errors):.2%}")
    print(f"Median Relative Error: {np.median(errors):.2%}")
    print(f"Mean Absolute Error: {np.mean(abs_errors):.0f} steps")
