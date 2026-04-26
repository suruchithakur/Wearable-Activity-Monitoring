import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.stats import skew, kurtosis


def _safe_get(recording, key):
    """Safely get sensor data, returns None if not available."""
    if key in recording.data:
        vals = recording.data[key].values
        try:
            vals = np.array(vals, dtype=float)
            # Replace any NaN/inf with 0
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            return vals
        except (ValueError, TypeError):
            return None
    return None


def _bandpass_filter(signal, fs, low=0.5, high=10.0, order=4):
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


def _time_domain_features(signal, prefix):
    """Extract comprehensive time-domain features from a signal."""
    features = {}
    if signal is None or len(signal) == 0:
        return features

    features[f'{prefix}_mean'] = np.mean(signal)
    features[f'{prefix}_std'] = np.std(signal)
    features[f'{prefix}_var'] = np.var(signal)
    features[f'{prefix}_rms'] = np.sqrt(np.mean(signal**2))
    features[f'{prefix}_max'] = np.max(signal)
    features[f'{prefix}_min'] = np.min(signal)
    features[f'{prefix}_range'] = np.max(signal) - np.min(signal)
    features[f'{prefix}_median'] = np.median(signal)
    features[f'{prefix}_iqr'] = np.percentile(signal, 75) - np.percentile(signal, 25)
    features[f'{prefix}_p10'] = np.percentile(signal, 10)
    features[f'{prefix}_p90'] = np.percentile(signal, 90)
    features[f'{prefix}_skew'] = skew(signal)
    features[f'{prefix}_kurtosis'] = kurtosis(signal)
    features[f'{prefix}_energy'] = np.sum(signal**2) / len(signal)
    features[f'{prefix}_abs_mean'] = np.mean(np.abs(signal))

    # Zero-crossing rate
    detrended = signal - np.mean(signal)
    features[f'{prefix}_zcr'] = np.mean(np.diff(np.sign(detrended)) != 0)

    # Mean-crossing rate
    features[f'{prefix}_mcr'] = np.mean(np.diff(np.sign(signal - np.mean(signal))) != 0)

    return features


def _freq_domain_features(signal, fs, prefix):
    """Extract frequency-domain features from a signal."""
    features = {}
    if signal is None or len(signal) < 64 or fs <= 0:
        return features

    n = len(signal)
    # Remove DC offset
    signal_centered = signal - np.mean(signal)

    # FFT
    f_vals = np.abs(fft(signal_centered)[:n // 2])
    freqs = np.linspace(0, fs / 2, n // 2)

    if len(f_vals) == 0 or np.sum(f_vals) == 0:
        return features

    # Dominant frequency
    features[f'{prefix}_dom_freq'] = freqs[np.argmax(f_vals)]

    # Spectral energy in bands
    total_energy = np.sum(f_vals**2)
    bands = {
        'sub_1hz': (0.0, 1.0),
        'walk_band': (0.5, 3.0),
        'run_band': (3.0, 8.0),
        'high_band': (8.0, 15.0),
    }
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        energy = np.sum(f_vals[mask]**2) if np.any(mask) else 0
        features[f'{prefix}_energy_{band_name}'] = energy
        features[f'{prefix}_ratio_{band_name}'] = energy / (total_energy + 1e-10)

    # Spectral centroid
    features[f'{prefix}_spec_centroid'] = np.sum(freqs * f_vals) / (np.sum(f_vals) + 1e-10)

    # Spectral entropy
    psd = f_vals**2
    psd_norm = psd / (np.sum(psd) + 1e-10)
    psd_norm = psd_norm[psd_norm > 0]
    features[f'{prefix}_spec_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    # Welch PSD for more robust frequency estimation
    try:
        nperseg = min(256, len(signal) // 2)
        if nperseg > 8:
            f_welch, pxx = welch(signal_centered, fs=fs, nperseg=nperseg)
            features[f'{prefix}_welch_dom_freq'] = f_welch[np.argmax(pxx)]
            features[f'{prefix}_welch_peak_power'] = np.max(pxx)
    except Exception:
        pass

    return features


def _windowed_features(signal, fs, prefix, window_sec=5.0):
    """Extract features from sliding windows to capture temporal dynamics."""
    features = {}
    if signal is None or len(signal) == 0 or fs <= 0:
        return features

    window_size = int(window_sec * fs)
    if window_size < 10 or len(signal) < window_size:
        return features

    # Compute per-window statistics
    n_windows = len(signal) // window_size
    if n_windows < 2:
        return features

    window_stds = []
    window_energies = []
    window_ranges = []
    window_zcrs = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        w = signal[start:end]
        window_stds.append(np.std(w))
        window_energies.append(np.sum(w**2) / len(w))
        window_ranges.append(np.max(w) - np.min(w))
        detrended = w - np.mean(w)
        window_zcrs.append(np.mean(np.diff(np.sign(detrended)) != 0))

    window_stds = np.array(window_stds)
    window_energies = np.array(window_energies)

    # Statistics of windowed features
    features[f'{prefix}_win_std_mean'] = np.mean(window_stds)
    features[f'{prefix}_win_std_std'] = np.std(window_stds)
    features[f'{prefix}_win_std_max'] = np.max(window_stds)
    features[f'{prefix}_win_std_min'] = np.min(window_stds)
    features[f'{prefix}_win_energy_mean'] = np.mean(window_energies)
    features[f'{prefix}_win_energy_std'] = np.std(window_energies)
    features[f'{prefix}_win_range_mean'] = np.mean(window_ranges)
    features[f'{prefix}_win_range_std'] = np.std(window_ranges)
    features[f'{prefix}_win_zcr_mean'] = np.mean(window_zcrs)
    features[f'{prefix}_win_zcr_std'] = np.std(window_zcrs)

    # Activity level segmentation (fraction of windows in low/med/high activity)
    # Thresholds based on empirical observation of accelerometer magnitude std
    low_thresh = np.percentile(window_stds, 25)
    high_thresh = np.percentile(window_stds, 75)
    features[f'{prefix}_win_frac_low'] = np.mean(window_stds <= low_thresh)
    features[f'{prefix}_win_frac_high'] = np.mean(window_stds >= high_thresh)

    # Coefficient of variation of windowed energy (high = multiple activity types)
    features[f'{prefix}_win_energy_cv'] = np.std(window_energies) / (np.mean(window_energies) + 1e-10)

    return features


def _jerk_features(signal, fs, prefix):
    """Extract jerk (derivative) features."""
    features = {}
    if signal is None or len(signal) < 10 or fs <= 0:
        return features

    jerk = np.diff(signal) * fs
    features[f'{prefix}_jerk_mean'] = np.mean(np.abs(jerk))
    features[f'{prefix}_jerk_std'] = np.std(jerk)
    features[f'{prefix}_jerk_max'] = np.max(np.abs(jerk))

    return features


def _correlation_features(sig1, sig2, prefix):
    """Compute correlation between two signals."""
    features = {}
    if sig1 is None or sig2 is None:
        return features
    min_len = min(len(sig1), len(sig2))
    if min_len < 10:
        return features
    features[f'{prefix}_corr'] = np.corrcoef(sig1[:min_len], sig2[:min_len])[0, 1]
    return features


def _autocorrelation_features(signal, fs, prefix):
    """Extract autocorrelation-based features for cadence detection."""
    features = {}
    if signal is None or len(signal) < 100 or fs <= 0:
        return features

    # Normalize
    sig = signal - np.mean(signal)
    if np.std(sig) < 1e-10:
        return features
    sig = sig / np.std(sig)

    # Compute autocorrelation for lags corresponding to 0.3s to 2.0s (step periods)
    min_lag = int(0.3 * fs)
    max_lag = min(int(2.0 * fs), len(sig) // 2)

    if min_lag >= max_lag:
        return features

    autocorr = np.correlate(sig, sig, mode='full')
    autocorr = autocorr[len(sig) - 1:]  # Keep positive lags
    autocorr = autocorr / autocorr[0]   # Normalize

    search_region = autocorr[min_lag:max_lag]
    if len(search_region) == 0:
        return features

    # Find first significant peak
    peaks, props = find_peaks(search_region, height=0.1)
    if len(peaks) > 0:
        best_peak = peaks[np.argmax(props['peak_heights'])]
        features[f'{prefix}_autocorr_period'] = (best_peak + min_lag) / fs
        features[f'{prefix}_autocorr_strength'] = props['peak_heights'][np.argmax(props['peak_heights'])]
    else:
        features[f'{prefix}_autocorr_period'] = 0
        features[f'{prefix}_autocorr_strength'] = 0

    return features


def extract_features(recording):
    """Extract comprehensive features from a recording for classification."""
    features = {}

    # ============================================================
    # 1. WATCH ACCELEROMETER (ax, ay, az)
    # ============================================================
    ax = _safe_get(recording, 'ax')
    ay = _safe_get(recording, 'ay')
    az = _safe_get(recording, 'az')

    if ax is not None and ay is not None and az is not None:
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        fs_acc = recording.data['ax'].samplerate

        # Time domain per-axis + magnitude
        for name, signal in [('ax', ax), ('ay', ay), ('az', az), ('acc_mag', mag)]:
            features.update(_time_domain_features(signal, name))
            features.update(_jerk_features(signal, fs_acc, name))

        # SMA (Signal Magnitude Area) — key HAR feature
        features['acc_sma'] = (np.sum(np.abs(ax)) + np.sum(np.abs(ay)) + np.sum(np.abs(az))) / len(ax)

        # Tilt angle features (gravity direction indicates body placement)
        mean_ax, mean_ay, mean_az = np.mean(ax), np.mean(ay), np.mean(az)
        mean_mag_val = np.sqrt(mean_ax**2 + mean_ay**2 + mean_az**2) + 1e-10
        features['acc_tilt_x'] = np.arccos(np.clip(mean_ax / mean_mag_val, -1, 1))
        features['acc_tilt_y'] = np.arccos(np.clip(mean_ay / mean_mag_val, -1, 1))
        features['acc_tilt_z'] = np.arccos(np.clip(mean_az / mean_mag_val, -1, 1))

        # Inter-axis correlations
        features.update(_correlation_features(ax, ay, 'acc_xy'))
        features.update(_correlation_features(ax, az, 'acc_xz'))
        features.update(_correlation_features(ay, az, 'acc_yz'))

        # Frequency domain features
        features.update(_freq_domain_features(mag, fs_acc, 'acc_mag'))
        features.update(_freq_domain_features(ax, fs_acc, 'ax'))
        features.update(_freq_domain_features(ay, fs_acc, 'ay'))
        features.update(_freq_domain_features(az, fs_acc, 'az'))

        # Windowed features (temporal dynamics)
        features.update(_windowed_features(mag, fs_acc, 'acc_mag'))
        features.update(_windowed_features(ax, fs_acc, 'ax'))
        features.update(_windowed_features(ay, fs_acc, 'ay'))
        features.update(_windowed_features(az, fs_acc, 'az'))

        # Autocorrelation features (cadence)
        # Filter first for cleaner periodicity
        mag_filtered = _bandpass_filter(mag, fs_acc, low=0.5, high=5.0)
        features.update(_autocorrelation_features(mag_filtered, fs_acc, 'acc_mag'))

        # Activity segment detection via windowed energy
        window_sec = 5.0
        window_size = int(window_sec * fs_acc)
        if window_size > 0 and len(mag) >= window_size:
            n_win = len(mag) // window_size
            win_energies = []
            for i in range(n_win):
                w = mag[i * window_size:(i + 1) * window_size]
                win_energies.append(np.std(w))
            win_energies = np.array(win_energies)

            # Low energy = standing/cycling, High energy = walking/running
            standing_thresh = 0.3  # empirical: low movement
            walking_thresh = 1.5   # moderate movement
            running_thresh = 4.0   # high movement

            features['frac_standing_windows'] = np.mean(win_energies < standing_thresh)
            features['frac_walking_windows'] = np.mean(
                (win_energies >= standing_thresh) & (win_energies < running_thresh))
            features['frac_running_windows'] = np.mean(win_energies >= running_thresh)
            features['n_activity_transitions'] = np.sum(
                np.abs(np.diff((win_energies > walking_thresh).astype(int))))

        # Step-like peak features on filtered magnitude
        try:
            mag_filt = _bandpass_filter(mag, fs_acc, 0.5, 5.0)
            peaks_walk, _ = find_peaks(mag_filt, height=0.1, distance=int(0.3 * fs_acc))
            features['n_detected_peaks'] = len(peaks_walk)
            features['peak_rate_hz'] = len(peaks_walk) / (len(mag) / fs_acc) if len(mag) > 0 else 0

            peaks_run, _ = find_peaks(mag_filt, height=0.5, distance=int(0.2 * fs_acc))
            features['n_high_peaks'] = len(peaks_run)
        except Exception:
            features['n_detected_peaks'] = 0
            features['peak_rate_hz'] = 0
            features['n_high_peaks'] = 0

    # ============================================================
    # 2. WATCH GYROSCOPE (gx, gy, gz)
    # ============================================================
    gx = _safe_get(recording, 'gx')
    gy = _safe_get(recording, 'gy')
    gz = _safe_get(recording, 'gz')

    if gx is not None and gy is not None and gz is not None:
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        fs_gyro = recording.data['gx'].samplerate

        for name, signal in [('gx', gx), ('gy', gy), ('gz', gz), ('gyro_mag', gyro_mag)]:
            features.update(_time_domain_features(signal, name))

        # Gyro frequency features (rotation patterns differ by watch location)
        features.update(_freq_domain_features(gyro_mag, fs_gyro, 'gyro_mag'))

        # Gyro correlations
        features.update(_correlation_features(gx, gy, 'gyro_xy'))
        features.update(_correlation_features(gx, gz, 'gyro_xz'))
        features.update(_correlation_features(gy, gz, 'gyro_yz'))

        # Windowed gyro features
        features.update(_windowed_features(gyro_mag, fs_gyro, 'gyro_mag'))

        # Total rotation energy
        features['gyro_total_energy'] = (np.sum(gx**2) + np.sum(gy**2) + np.sum(gz**2)) / len(gx)

    # ============================================================
    # 3. WATCH MAGNETOMETER (mx, my, mz)
    # ============================================================
    mx_val = _safe_get(recording, 'mx')
    my_val = _safe_get(recording, 'my')
    mz_val = _safe_get(recording, 'mz')

    if mx_val is not None and my_val is not None and mz_val is not None:
        mag_m = np.sqrt(mx_val**2 + my_val**2 + mz_val**2)

        for name, signal in [('mx', mx_val), ('my', my_val), ('mz', mz_val), ('mag_field', mag_m)]:
            features.update(_time_domain_features(signal, name))

        # Heading angle and changes (useful for path classification)
        heading = np.arctan2(my_val, mx_val)
        features['heading_mean'] = np.mean(heading)
        features['heading_std'] = np.std(heading)
        heading_changes = np.abs(np.diff(heading))
        # Handle wraparound
        heading_changes = np.minimum(heading_changes, 2 * np.pi - heading_changes)
        features['heading_total_change'] = np.sum(heading_changes)
        features['heading_change_rate'] = np.mean(heading_changes)

    # ============================================================
    # 4. PHONE SENSORS (allowed: gravity, orientation, pressure, linear accel, rotation)
    # ============================================================

    # Phone gravity (indicates phone orientation / body posture)
    pgx = _safe_get(recording, 'phone_gravx')
    pgy = _safe_get(recording, 'phone_gravy')
    pgz = _safe_get(recording, 'phone_gravz')
    if pgx is not None and pgy is not None and pgz is not None:
        for name, signal in [('pgravx', pgx), ('pgravy', pgy), ('pgravz', pgz)]:
            features[f'{name}_mean'] = np.mean(signal)
            features[f'{name}_std'] = np.std(signal)

    # Phone orientation
    for axis in ['phone_orientationx', 'phone_orientationy', 'phone_orientationz']:
        val = _safe_get(recording, axis)
        if val is not None:
            short = axis.replace('phone_', 'p')
            features[f'{short}_mean'] = np.mean(val)
            features[f'{short}_std'] = np.std(val)

    # Phone pressure (barometer — correlates with altitude, key for path direction)
    ppressure = _safe_get(recording, 'phone_pressure')
    if ppressure is not None and len(ppressure) > 3:
        features['ppressure_mean'] = np.mean(ppressure)
        features['ppressure_std'] = np.std(ppressure)
        features['ppressure_range'] = np.max(ppressure) - np.min(ppressure)
        t = np.linspace(0, 1, len(ppressure))
        # Pressure trend (negative = going uphill, positive = going downhill)
        p1 = np.polyfit(t, ppressure, 1)
        features['ppressure_slope'] = p1[0]
        features['ppressure_start'] = ppressure[0]
        features['ppressure_end'] = ppressure[-1]
        features['ppressure_delta'] = ppressure[-1] - ppressure[0]
        # Curvature
        p2 = np.polyfit(t, ppressure, 2)
        features['ppressure_curv'] = p2[0]
        # Half slopes
        mid = len(ppressure) // 2
        if mid > 3:
            th = np.linspace(0, 1, mid)
            features['ppressure_slope_first'] = np.polyfit(th, ppressure[:mid], 1)[0]
            features['ppressure_slope_second'] = np.polyfit(th, ppressure[mid:mid+mid], 1)[0]

    # Phone linear acceleration (phone motion without gravity)
    plax = _safe_get(recording, 'phone_lax')
    play = _safe_get(recording, 'phone_lay')
    plaz = _safe_get(recording, 'phone_laz')
    if plax is not None and play is not None and plaz is not None:
        min_len = min(len(plax), len(play), len(plaz))
        pla_mag = np.sqrt(plax[:min_len]**2 + play[:min_len]**2 + plaz[:min_len]**2)
        features.update(_time_domain_features(pla_mag, 'pla_mag'))

    # Phone accelerometer
    pax = _safe_get(recording, 'phone_ax')
    pay = _safe_get(recording, 'phone_ay')
    paz = _safe_get(recording, 'phone_az')
    if pax is not None and pay is not None and paz is not None:
        min_len = min(len(pax), len(pay), len(paz))
        pa_mag = np.sqrt(pax[:min_len]**2 + pay[:min_len]**2 + paz[:min_len]**2)
        features.update(_time_domain_features(pa_mag, 'pa_mag'))
        fs_pa = recording.data['phone_ax'].samplerate
        features.update(_freq_domain_features(pa_mag, fs_pa, 'pa_mag'))

    # Phone gyroscope
    pgx_g = _safe_get(recording, 'phone_gx')
    pgy_g = _safe_get(recording, 'phone_gy')
    pgz_g = _safe_get(recording, 'phone_gz')
    if pgx_g is not None and pgy_g is not None and pgz_g is not None:
        min_len = min(len(pgx_g), len(pgy_g), len(pgz_g))
        pg_mag = np.sqrt(pgx_g[:min_len]**2 + pgy_g[:min_len]**2 + pgz_g[:min_len]**2)
        features.update(_time_domain_features(pg_mag, 'pg_mag'))

    # ============================================================
    # 5. ALTITUDE & TEMPERATURE
    # ============================================================
    altitude = _safe_get(recording, 'altitude')
    if altitude is not None and len(altitude) > 3:
        features['altitude_mean'] = np.mean(altitude)
        features['altitude_std'] = np.std(altitude)
        features['altitude_range'] = np.max(altitude) - np.min(altitude)
        features['altitude_start'] = altitude[0]
        features['altitude_end'] = altitude[-1]
        features['altitude_delta'] = altitude[-1] - altitude[0]
        features['altitude_min'] = np.min(altitude)
        features['altitude_max'] = np.max(altitude)
        # Linear slope (uphill vs downhill → path classifier)
        t = np.linspace(0, 1, len(altitude))
        p1 = np.polyfit(t, altitude, 1)
        features['altitude_slope'] = p1[0]
        # Quadratic curvature (profile shape: concave vs convex)
        p2 = np.polyfit(t, altitude, 2)
        features['altitude_curv'] = p2[0]   # + = U-shape, - = hill-shape
        features['altitude_poly2_b'] = p2[1]  # linear term in quadratic fit
        # Segmented slopes: first-half vs second-half
        mid = len(altitude) // 2
        t_half = np.linspace(0, 1, mid)
        if mid > 3:
            features['altitude_slope_first'] = np.polyfit(t_half, altitude[:mid], 1)[0]
            features['altitude_slope_second'] = np.polyfit(t_half, altitude[mid:mid+mid], 1)[0]
            features['altitude_slope_diff'] = features['altitude_slope_second'] - features['altitude_slope_first']
        # Quartile segments
        q = len(altitude) // 4
        if q > 1:
            features['altitude_q1_mean'] = np.mean(altitude[:q])
            features['altitude_q2_mean'] = np.mean(altitude[q:2*q])
            features['altitude_q3_mean'] = np.mean(altitude[2*q:3*q])
            features['altitude_q4_mean'] = np.mean(altitude[3*q:])
            features['altitude_q1_to_q4'] = np.mean(altitude[3*q:]) - np.mean(altitude[:q])
        # Trend consistency: std of windowed slopes
        n_segs = 5
        seg_size = len(altitude) // n_segs
        if seg_size > 1:
            seg_slopes = []
            for i in range(n_segs):
                seg = altitude[i*seg_size:(i+1)*seg_size]
                seg_t = np.linspace(0, 1, len(seg))
                seg_slopes.append(np.polyfit(seg_t, seg, 1)[0])
            features['altitude_slope_consistency'] = np.std(seg_slopes)
            features['altitude_slope_segments_mean'] = np.mean(seg_slopes)
    else:
        for k in ['altitude_mean', 'altitude_std', 'altitude_range', 'altitude_start',
                  'altitude_end', 'altitude_delta', 'altitude_min', 'altitude_max',
                  'altitude_slope', 'altitude_curv']:
            features[k] = np.nan

    temperature = _safe_get(recording, 'temperature')
    if temperature is not None and len(temperature) > 0:
        features['temperature_mean'] = np.mean(temperature)
        features['temperature_std'] = np.std(temperature)
    else:
        features['temperature_mean'] = np.nan

    # ============================================================
    # 6. RECORDING-LEVEL FEATURES
    # ============================================================
    if 'ax' in recording.data:
        features['duration_s'] = recording.data['ax'].total_time
        features['acc_samplerate'] = recording.data['ax'].samplerate
    else:
        features['duration_s'] = 0

    return features


if __name__ == "__main__":
    import os
    import glob
    import ast
    from mhealth_activity import Recording

    metadata = pd.read_csv('train_metadata.csv')
    train_dir = 'data/train'

    all_features = []
    print(f"Extracting features for {len(metadata)} files...")

    for i, row in metadata.iterrows():
        if i % 50 == 0:
            print(f"Progress: {i}/{len(metadata)}")

        f_path = os.path.join(train_dir, row['filename'])
        try:
            rec = Recording(f_path)
            feats = extract_features(rec)
            feats['filename'] = row['filename']
            all_features.append(feats)
        except Exception as e:
            print(f"Error in {row['filename']}: {e}")

    df_features = pd.DataFrame(all_features)

    # Parse multi-label activities from recordings (handle dict and list formats)
    has_standing = []
    has_walking = []
    has_running = []
    has_cycling = []

    for _, row in metadata.iterrows():
        f_path = os.path.join(train_dir, row['filename'])
        try:
            rec = Recording(f_path)
            acts = rec.labels.get('activities', {})
            if isinstance(acts, dict):
                has_standing.append(acts.get('standing', False))
                has_walking.append(acts.get('walking', False))
                has_running.append(acts.get('running', False))
                has_cycling.append(acts.get('cycling', False))
            elif isinstance(acts, list):
                unique_acts = set(acts)
                has_standing.append(0 in unique_acts)
                has_walking.append(1 in unique_acts)
                has_running.append(2 in unique_acts)
                has_cycling.append(3 in unique_acts)
            else:
                has_standing.append(False)
                has_walking.append(False)
                has_running.append(False)
                has_cycling.append(False)
        except Exception:
            has_standing.append(False)
            has_walking.append(False)
            has_running.append(False)
            has_cycling.append(False)

    metadata['has_standing'] = has_standing
    metadata['has_walking'] = has_walking
    metadata['has_running'] = has_running
    metadata['has_cycling'] = has_cycling

    df_final = pd.merge(metadata, df_features, on='filename')
    df_final.to_csv('train_features.csv', index=False)
    print(f"Done! Saved {len(df_final)} rows with {len(df_features.columns)} features to train_features.csv")
    print(f"Multi-label columns: has_standing, has_walking, has_running, has_cycling")
    print(f"\nLabel distribution:")
    print(f"  Standing: {sum(has_standing)}")
    print(f"  Walking: {sum(has_walking)}")
    print(f"  Running: {sum(has_running)}")
    print(f"  Cycling: {sum(has_cycling)}")
