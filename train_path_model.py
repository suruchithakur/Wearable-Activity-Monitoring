# conda activate mhealth26 && python train_path_model.py

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import pickle

from mhealth_activity import Recording


# ============================================================
# Trace helpers
# ============================================================

def trace_values(recording, key):
    if key not in recording.data:
        return np.array([], dtype=float)
    tr = recording.data[key]
    for attr in ["values", "data", "y", "v"]:
        if hasattr(tr, attr):
            arr = np.asarray(getattr(tr, attr), dtype=float)
            return arr[np.isfinite(arr)]
    try:
        arr = np.asarray(tr, dtype=float)
        return arr[np.isfinite(arr)]
    except Exception:
        raise ValueError(f"Could not extract numeric values from {key}")


def trace_total_time(recording, key="timestamp"):
    if key not in recording.data:
        return np.nan
    return getattr(recording.data[key], "total_time", np.nan)


def magnitude(x, y, z):
    n = min(len(x), len(y), len(z))
    if n == 0:
        return np.array([], dtype=float)
    return np.sqrt(x[:n] ** 2 + y[:n] ** 2 + z[:n] ** 2)


# ============================================================
# Generic stat helpers
# ============================================================

def basic_stats(x, prefix):
    if len(x) < 5:
        return {
            f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan, f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan, f"{prefix}_median": np.nan,
            f"{prefix}_iqr": np.nan, f"{prefix}_skew": np.nan,
            f"{prefix}_kurt": np.nan,
        }
    return {
        f"{prefix}_mean": np.mean(x), f"{prefix}_std": np.std(x),
        f"{prefix}_min": np.min(x), f"{prefix}_max": np.max(x),
        f"{prefix}_range": np.max(x) - np.min(x),
        f"{prefix}_median": np.median(x),
        f"{prefix}_iqr": np.percentile(x, 75) - np.percentile(x, 25),
        f"{prefix}_skew": skew(x), f"{prefix}_kurt": kurtosis(x),
    }


def peak_features(x, prefix):
    if len(x) < 20:
        return {f"{prefix}_n_peaks": np.nan, f"{prefix}_peak_rate": np.nan,
                f"{prefix}_peak_prom_mean": np.nan}
    x_centered = x - np.median(x)
    prom = max(np.std(x_centered), 1e-6)
    peaks, props = find_peaks(x_centered, prominence=prom, distance=5)
    if len(peaks) == 0:
        return {f"{prefix}_n_peaks": 0, f"{prefix}_peak_rate": 0,
                f"{prefix}_peak_prom_mean": 0}
    return {
        f"{prefix}_n_peaks": len(peaks),
        f"{prefix}_peak_rate": len(peaks) / len(x),
        f"{prefix}_peak_prom_mean": np.mean(props["prominences"]),
    }


def segment_stats(x, prefix, n_segments=8):
    feats = {}
    for i in range(n_segments):
        feats[f"{prefix}_seg{i}_mean"] = np.nan
        feats[f"{prefix}_seg{i}_std"] = np.nan
        feats[f"{prefix}_seg{i}_range"] = np.nan
        feats[f"{prefix}_seg{i}_median"] = np.nan
    if len(x) < n_segments * 5:
        return feats
    for i, seg in enumerate(np.array_split(x, n_segments)):
        feats[f"{prefix}_seg{i}_mean"] = np.mean(seg)
        feats[f"{prefix}_seg{i}_std"] = np.std(seg)
        feats[f"{prefix}_seg{i}_range"] = np.max(seg) - np.min(seg)
        feats[f"{prefix}_seg{i}_median"] = np.median(seg)
    return feats


def burst_features(x, prefix):
    if len(x) < 20:
        return {f"{prefix}_burst_count": np.nan, f"{prefix}_burst_ratio": np.nan,
                f"{prefix}_burst_mean_value": np.nan}
    threshold = np.percentile(x, 90)
    mask = x > threshold
    return {
        f"{prefix}_burst_count": np.sum(np.diff(mask.astype(int)) == 1),
        f"{prefix}_burst_ratio": np.mean(mask),
        f"{prefix}_burst_mean_value": np.mean(x[mask]) if np.any(mask) else 0,
    }


# ============================================================
# Heading / turn helpers
# ============================================================

def heading_array(x_mag, y_mag):
    n = min(len(x_mag), len(y_mag))
    if n < 10:
        return np.array([], dtype=float)
    return np.unwrap(np.arctan2(y_mag[:n], x_mag[:n]))


def normalize_heading(h):
    if len(h) == 0:
        return h
    return h - h[0]


def downsample_signal(x, n_points=20):
    if len(x) < 10:
        return [np.nan] * n_points
    idx = np.linspace(0, len(x) - 1, n_points).astype(int)
    return x[idx]


def heading_features(heading, prefix):
    if len(heading) < 10:
        return {
            f"{prefix}_total_abs_change": np.nan, f"{prefix}_net_change": np.nan,
            f"{prefix}_std": np.nan, f"{prefix}_start": np.nan, f"{prefix}_end": np.nan,
        }
    diff = np.diff(heading)
    return {
        f"{prefix}_total_abs_change": np.sum(np.abs(diff)),
        f"{prefix}_net_change": heading[-1] - heading[0],
        f"{prefix}_std": np.std(heading),
        f"{prefix}_start": heading[0],
        f"{prefix}_end": heading[-1],
    }


def segment_net_change(x, prefix, n_segments=8):
    feats = {}
    for i in range(n_segments):
        feats[f"{prefix}_seg{i}_net_change"] = np.nan
        feats[f"{prefix}_seg{i}_total_abs_change"] = np.nan
        feats[f"{prefix}_seg{i}_signed_pos_sum"] = np.nan
        feats[f"{prefix}_seg{i}_signed_neg_sum"] = np.nan
    if len(x) < n_segments * 5:
        return feats
    for i, seg in enumerate(np.array_split(x, n_segments)):
        diff = np.diff(seg)
        feats[f"{prefix}_seg{i}_net_change"] = seg[-1] - seg[0]
        feats[f"{prefix}_seg{i}_total_abs_change"] = np.sum(np.abs(diff))
        feats[f"{prefix}_seg{i}_signed_pos_sum"] = np.sum(diff[diff > 0])
        feats[f"{prefix}_seg{i}_signed_neg_sum"] = np.sum(diff[diff < 0])
    return feats


def signed_turn_features(signal, prefix, n_segments=8):
    feats = {}
    for k in ["pos_turn_count", "neg_turn_count", "pos_turn_sum", "neg_turn_sum",
              "turn_balance", "abs_turn_sum", "large_turn_count"]:
        feats[f"{prefix}_{k}"] = np.nan
    for i in range(n_segments):
        feats[f"{prefix}_seg{i}_pos_turn_count"] = np.nan
        feats[f"{prefix}_seg{i}_neg_turn_count"] = np.nan
        feats[f"{prefix}_seg{i}_pos_turn_sum"] = np.nan
        feats[f"{prefix}_seg{i}_neg_turn_sum"] = np.nan
        feats[f"{prefix}_seg{i}_turn_balance"] = np.nan
    if len(signal) < 20:
        return feats
    diff = np.diff(signal)
    abs_diff = np.abs(diff)
    threshold = np.percentile(abs_diff, 90)
    pos = diff > threshold
    neg = diff < -threshold
    pos_sum = np.sum(diff[pos]) if np.any(pos) else 0
    neg_sum = np.sum(diff[neg]) if np.any(neg) else 0
    feats[f"{prefix}_pos_turn_count"] = np.sum(pos)
    feats[f"{prefix}_neg_turn_count"] = np.sum(neg)
    feats[f"{prefix}_pos_turn_sum"] = pos_sum
    feats[f"{prefix}_neg_turn_sum"] = neg_sum
    feats[f"{prefix}_turn_balance"] = pos_sum + neg_sum
    feats[f"{prefix}_abs_turn_sum"] = np.sum(abs_diff)
    feats[f"{prefix}_large_turn_count"] = np.sum(abs_diff > threshold)
    for i, seg in enumerate(np.array_split(signal, n_segments)):
        if len(seg) < 5:
            continue
        d = np.diff(seg)
        abs_d = np.abs(d)
        th = np.percentile(abs_d, 90)
        p = d > th
        n = d < -th
        p_sum = np.sum(d[p]) if np.any(p) else 0
        n_sum = np.sum(d[n]) if np.any(n) else 0
        feats[f"{prefix}_seg{i}_pos_turn_count"] = np.sum(p)
        feats[f"{prefix}_seg{i}_neg_turn_count"] = np.sum(n)
        feats[f"{prefix}_seg{i}_pos_turn_sum"] = p_sum
        feats[f"{prefix}_seg{i}_neg_turn_sum"] = n_sum
        feats[f"{prefix}_seg{i}_turn_balance"] = p_sum + n_sum
    return feats


def gyro_signed_features(axis_signal, prefix, n_segments=8):
    feats = {}
    for key in ["positive_energy", "negative_energy", "signed_energy_balance",
                "positive_ratio", "negative_ratio", "large_positive_count", "large_negative_count"]:
        feats[f"{prefix}_{key}"] = np.nan
    for i in range(n_segments):
        feats[f"{prefix}_seg{i}_signed_sum"] = np.nan
        feats[f"{prefix}_seg{i}_abs_sum"] = np.nan
        feats[f"{prefix}_seg{i}_positive_ratio"] = np.nan
        feats[f"{prefix}_seg{i}_negative_ratio"] = np.nan
    if len(axis_signal) < 20:
        return feats
    x = axis_signal
    pos = x[x > 0]
    neg = x[x < 0]
    feats[f"{prefix}_positive_energy"] = np.mean(pos ** 2) if len(pos) else 0
    feats[f"{prefix}_negative_energy"] = np.mean(neg ** 2) if len(neg) else 0
    feats[f"{prefix}_signed_energy_balance"] = (
        feats[f"{prefix}_positive_energy"] - feats[f"{prefix}_negative_energy"]
    )
    feats[f"{prefix}_positive_ratio"] = np.mean(x > 0)
    feats[f"{prefix}_negative_ratio"] = np.mean(x < 0)
    threshold = np.percentile(np.abs(x), 90)
    feats[f"{prefix}_large_positive_count"] = np.sum(x > threshold)
    feats[f"{prefix}_large_negative_count"] = np.sum(x < -threshold)
    for i, seg in enumerate(np.array_split(x, n_segments)):
        feats[f"{prefix}_seg{i}_signed_sum"] = np.sum(seg)
        feats[f"{prefix}_seg{i}_abs_sum"] = np.sum(np.abs(seg))
        feats[f"{prefix}_seg{i}_positive_ratio"] = np.mean(seg > 0)
        feats[f"{prefix}_seg{i}_negative_ratio"] = np.mean(seg < 0)
    return feats


# ============================================================
# Main feature extraction (full recording)
# ============================================================

def extract_path_features(recording, n_segments=8):
    features = {}
    features["duration_sec"] = trace_total_time(recording, "timestamp")

    altitude = trace_values(recording, "altitude")
    if len(altitude) >= 10:
        initial_alt = np.median(altitude[:10])
        final_alt = np.median(altitude[-10:])
        alt_delta = final_alt - initial_alt
        features["alt_initial"] = initial_alt
        features["alt_final"] = final_alt
        features["alt_delta"] = alt_delta
        features["is_uphill"] = int(alt_delta > 0)
    else:
        features["alt_initial"] = np.nan
        features["alt_final"] = np.nan
        features["alt_delta"] = np.nan
        features["is_uphill"] = np.nan
    features.update(segment_stats(altitude, "altitude", n_segments=n_segments))

    gx = trace_values(recording, "gx")
    gy = trace_values(recording, "gy")
    gz = trace_values(recording, "gz")
    gyro_mag = magnitude(gx, gy, gz)
    features.update(basic_stats(gyro_mag, "gyro_mag"))
    features.update(peak_features(gyro_mag, "gyro_mag"))
    features.update(burst_features(gyro_mag, "gyro_mag"))
    features.update(segment_stats(gyro_mag, "gyro_mag", n_segments=n_segments))
    for axis_name, axis_signal in [("gx", gx), ("gy", gy), ("gz", gz)]:
        features.update(basic_stats(axis_signal, axis_name))
        features.update(segment_stats(axis_signal, axis_name, n_segments=n_segments))
        features.update(gyro_signed_features(axis_signal, axis_name, n_segments=n_segments))

    pgx = trace_values(recording, "phone_gx")
    pgy = trace_values(recording, "phone_gy")
    pgz = trace_values(recording, "phone_gz")
    phone_gyro_mag = magnitude(pgx, pgy, pgz)
    features.update(basic_stats(phone_gyro_mag, "phone_gyro_mag"))
    features.update(peak_features(phone_gyro_mag, "phone_gyro_mag"))
    features.update(burst_features(phone_gyro_mag, "phone_gyro_mag"))
    features.update(segment_stats(phone_gyro_mag, "phone_gyro_mag", n_segments=n_segments))
    for axis_name, axis_signal in [("phone_gx", pgx), ("phone_gy", pgy), ("phone_gz", pgz)]:
        features.update(basic_stats(axis_signal, axis_name))
        features.update(segment_stats(axis_signal, axis_name, n_segments=n_segments))
        features.update(gyro_signed_features(axis_signal, axis_name, n_segments=n_segments))

    mx = trace_values(recording, "mx")
    my = trace_values(recording, "my")
    mz = trace_values(recording, "mz")
    mag_mag = magnitude(mx, my, mz)
    watch_heading = heading_array(mx, my)
    features.update(basic_stats(mag_mag, "watch_mag_mag"))
    features.update(segment_stats(mag_mag, "watch_mag_mag", n_segments=n_segments))
    for axis_name, axis_signal in [("mx", mx), ("my", my), ("mz", mz)]:
        features.update(basic_stats(axis_signal, axis_name))
        features.update(segment_stats(axis_signal, axis_name, n_segments=n_segments))
    features.update(heading_features(watch_heading, "watch_heading"))
    features.update(segment_stats(watch_heading, "watch_heading", n_segments=n_segments))
    features.update(segment_net_change(watch_heading, "watch_heading", n_segments=n_segments))
    features.update(signed_turn_features(watch_heading, "watch_heading_turn", n_segments=n_segments))

    phone_mx = trace_values(recording, "phone_mx")
    phone_my = trace_values(recording, "phone_my")
    phone_mz = trace_values(recording, "phone_mz")
    phone_mag_mag = magnitude(phone_mx, phone_my, phone_mz)
    phone_heading = heading_array(phone_mx, phone_my)
    features.update(basic_stats(phone_mag_mag, "phone_mag_mag"))
    features.update(segment_stats(phone_mag_mag, "phone_mag_mag", n_segments=n_segments))
    for axis_name, axis_signal in [("phone_mx", phone_mx), ("phone_my", phone_my), ("phone_mz", phone_mz)]:
        features.update(basic_stats(axis_signal, axis_name))
        features.update(segment_stats(axis_signal, axis_name, n_segments=n_segments))
    features.update(heading_features(phone_heading, "phone_heading"))
    features.update(segment_stats(phone_heading, "phone_heading", n_segments=n_segments))
    features.update(segment_net_change(phone_heading, "phone_heading", n_segments=n_segments))
    features.update(signed_turn_features(phone_heading, "phone_heading_turn", n_segments=n_segments))

    ax = trace_values(recording, "ax")
    ay = trace_values(recording, "ay")
    az = trace_values(recording, "az")
    acc_mag = magnitude(ax, ay, az)
    features.update(segment_stats(acc_mag, "acc_mag", n_segments=6))
    features.update(burst_features(acc_mag, "acc_mag"))
    features.update(peak_features(acc_mag, "acc_mag"))

    return features


# ============================================================
# Prefix feature extraction (first fraction of recording)
# ============================================================

def prefix_slice(x, fraction=0.30):
    if len(x) < 10:
        return np.array([], dtype=float)
    n = max(10, int(len(x) * fraction))
    return x[:n]


def extract_prefix_features(recording, fraction=0.30):
    features = {}

    def pref(key):
        return prefix_slice(trace_values(recording, key), fraction=fraction)

    mx = pref("mx")
    my = pref("my")
    mz = pref("mz")
    features.update(basic_stats(mx, "mx_prefix"))
    features.update(basic_stats(my, "my_prefix"))
    features.update(basic_stats(mz, "mz_prefix"))

    watch_heading = normalize_heading(heading_array(mx, my))
    features.update(heading_features(watch_heading, "watch_heading_prefix"))
    features.update(signed_turn_features(watch_heading, "watch_heading_prefix_turn", n_segments=4))
    for i, val in enumerate(downsample_signal(watch_heading, n_points=20)):
        features[f"watch_heading_shape_{i}"] = val
    for i, val in enumerate(np.diff(downsample_signal(watch_heading, n_points=20))):
        features[f"watch_heading_shape_diff_{i}"] = val

    phone_mx = pref("phone_mx")
    phone_my = pref("phone_my")
    phone_mz = pref("phone_mz")
    features.update(basic_stats(phone_mx, "phone_mx_prefix"))
    features.update(basic_stats(phone_my, "phone_my_prefix"))
    features.update(basic_stats(phone_mz, "phone_mz_prefix"))

    phone_heading = normalize_heading(heading_array(phone_mx, phone_my))
    features.update(heading_features(phone_heading, "phone_heading_prefix"))
    features.update(signed_turn_features(phone_heading, "phone_heading_prefix_turn", n_segments=4))
    for i, val in enumerate(downsample_signal(phone_heading, n_points=20)):
        features[f"phone_heading_shape_{i}"] = val
    for i, val in enumerate(np.diff(downsample_signal(phone_heading, n_points=20))):
        features[f"phone_heading_shape_diff_{i}"] = val

    gx = pref("gx")
    gy = pref("gy")
    gz = pref("gz")
    features.update(basic_stats(gx, "gx_prefix"))
    features.update(basic_stats(gy, "gy_prefix"))
    features.update(basic_stats(gz, "gz_prefix"))
    gyro_mag = magnitude(gx, gy, gz)
    features.update(basic_stats(gyro_mag, "gyro_mag_prefix"))
    features.update(burst_features(gyro_mag, "gyro_mag_prefix"))

    phone_gx = pref("phone_gx")
    phone_gy = pref("phone_gy")
    phone_gz = pref("phone_gz")
    features.update(basic_stats(phone_gx, "phone_gx_prefix"))
    features.update(basic_stats(phone_gy, "phone_gy_prefix"))
    features.update(basic_stats(phone_gz, "phone_gz_prefix"))
    phone_gyro_mag = magnitude(phone_gx, phone_gy, phone_gz)
    features.update(basic_stats(phone_gyro_mag, "phone_gyro_mag_prefix"))
    features.update(burst_features(phone_gyro_mag, "phone_gyro_mag_prefix"))

    alt = pref("altitude")
    if len(alt) >= 10:
        features["alt_prefix_initial"] = np.median(alt[:5])
        features["alt_prefix_final"] = np.median(alt[-5:])
        features["alt_prefix_delta"] = np.median(alt[-5:]) - np.median(alt[:5])
        features["alt_prefix_range"] = np.max(alt) - np.min(alt)
        features["alt_prefix_std"] = np.std(alt)
    else:
        features["alt_prefix_initial"] = np.nan
        features["alt_prefix_final"] = np.nan
        features["alt_prefix_delta"] = np.nan
        features["alt_prefix_range"] = np.nan
        features["alt_prefix_std"] = np.nan

    return features


# ============================================================
# Model classes
# ============================================================

class TwoStagePathClassifier:
    def __init__(self):
        self.direction_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=500, random_state=1,
                class_weight="balanced", n_jobs=1,
            ))
        ])
        self.uphill_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=900, random_state=2,
                class_weight="balanced", n_jobs=1,
                max_features="sqrt",
            ))
        ])
        self.downhill_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=700, random_state=3,
                class_weight="balanced", n_jobs=1,
                max_features="sqrt",
            ))
        ])
        self.feature_columns = None

    def fit(self, X, y):
        self.feature_columns = list(X.columns)
        y_direction = np.isin(y, [0, 1, 2]).astype(int)
        self.direction_model.fit(X, y_direction)
        uphill_mask = np.isin(y, [0, 1, 2])
        downhill_mask = np.isin(y, [3, 4])
        self.uphill_model.fit(X[uphill_mask], y[uphill_mask])
        self.downhill_model.fit(X[downhill_mask], y[downhill_mask])
        return self

    def predict(self, X):
        X = X[self.feature_columns]
        direction_pred = self.direction_model.predict(X)
        preds = []
        for i in range(len(X)):
            row = X.iloc[[i]]
            if direction_pred[i] == 1:
                preds.append(self.uphill_model.predict(row)[0])
            else:
                preds.append(self.downhill_model.predict(row)[0])
        return np.array(preds)


class PrefixZeroVsTwoRefiner:
    def __init__(self, fraction=0.30):
        self.fraction = fraction
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=700, random_state=42,
                class_weight="balanced", n_jobs=1,
                max_features="sqrt",
            ))
        ])
        self.feature_columns = None

    def fit(self, recordings_subset, y_subset):
        rows = [extract_prefix_features(r, fraction=self.fraction) for r in recordings_subset]
        X_prefix = pd.DataFrame(rows)
        self.feature_columns = list(X_prefix.columns)
        mask = np.isin(y_subset, [0, 2])
        self.model.fit(X_prefix.loc[mask, self.feature_columns], y_subset[mask])
        return self

    def predict(self, recordings_subset):
        rows = [extract_prefix_features(r, fraction=self.fraction) for r in recordings_subset]
        X_prefix = pd.DataFrame(rows)[self.feature_columns]
        return self.model.predict(X_prefix)


class PathClassifier:
    """
    Single serializable model: TwoStagePathClassifier + PrefixZeroVsTwoRefiner.

    Usage:
        model = PathClassifier(prefix_fraction=0.30)
        model.fit(recordings, y)
        with open("path_model.pkl", "wb") as f: pickle.dump(model, f)

        with open("path_model.pkl", "rb") as f: model = pickle.load(f)
        path = model.predict_one(recording)
    """

    def __init__(self, prefix_fraction=0.30, n_segments=8):
        self.prefix_fraction = prefix_fraction
        self.n_segments = n_segments
        self._main = TwoStagePathClassifier()
        self._refiner = PrefixZeroVsTwoRefiner(fraction=prefix_fraction)

    def fit(self, recordings, y):
        print("  Extracting main features...")
        X = pd.DataFrame([extract_path_features(r, n_segments=self.n_segments) for r in recordings])
        print("  Training main two-stage model...")
        self._main.fit(X, y)
        print("  Training prefix 0-vs-2 refiner...")
        self._refiner.fit(recordings, y)
        return self

    def predict(self, recordings):
        X = pd.DataFrame([extract_path_features(r, n_segments=self.n_segments) for r in recordings])
        base = self._main.predict(X)
        refined = base.copy()
        idx = [i for i, p in enumerate(base) if p in [0, 2]]
        if idx:
            refined_02 = self._refiner.predict([recordings[i] for i in idx])
            for j, label in zip(idx, refined_02):
                refined[j] = label
        return refined

    def predict_one(self, recording):
        return int(self.predict([recording])[0])


# ============================================================
# Training script
# ============================================================

def main():
    TRAIN_DIR = "data/train"
    OUTPUT_MODEL = os.path.join("models", "path_model.pkl")
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    FRACTIONS_TO_TRY = [0.15, 0.20, 0.25, 0.30, 0.35]

    # ----------------------------------------------------------
    # Load recordings
    # ----------------------------------------------------------
    print("Loading recordings...")
    recordings, path_labels = [], []
    for filename in sorted(os.listdir(TRAIN_DIR)):
        if not filename.endswith(".pkl"):
            continue
        rec = Recording(os.path.join(TRAIN_DIR, filename))
        if rec.labels is None:
            continue
        recordings.append(rec)
        path_labels.append(rec.labels["path_idx"])

    y = np.array(path_labels)
    print(f"Loaded {len(recordings)} recordings")
    print("Path label counts:")
    print(pd.Series(y).value_counts().sort_index().to_string())

    # ----------------------------------------------------------
    # Extract main features (once for all recordings)
    # ----------------------------------------------------------
    print("\nExtracting main features...")
    X = pd.DataFrame([extract_path_features(r, n_segments=8) for r in recordings])
    print(f"Feature matrix: {X.shape}")

    # ----------------------------------------------------------
    # Train / validation split
    # ----------------------------------------------------------
    indices = np.arange(len(recordings))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    recs_train = [recordings[i] for i in train_idx]
    recs_val = [recordings[i] for i in val_idx]

    # ----------------------------------------------------------
    # Fit main model on training split
    # ----------------------------------------------------------
    print("\nFitting main two-stage model...")
    main_model = TwoStagePathClassifier()
    main_model.fit(X_train, y_train)
    base_pred = main_model.predict(X_val)
    print(f"Base accuracy (no refiner): {accuracy_score(y_val, base_pred):.4f}")

    # ----------------------------------------------------------
    # Auto-select best prefix fraction
    # ----------------------------------------------------------
    print("\nSearching for best prefix fraction...")
    best_fraction, best_acc = None, -1.0
    for frac in FRACTIONS_TO_TRY:
        refiner = PrefixZeroVsTwoRefiner(fraction=frac)
        refiner.fit(recs_train, y_train)

        refined = base_pred.copy()
        candidate_idx = [i for i, p in enumerate(base_pred) if p in [0, 2]]
        if candidate_idx:
            refined_02 = refiner.predict([recs_val[i] for i in candidate_idx])
            for j, label in zip(candidate_idx, refined_02):
                refined[j] = label

        acc = accuracy_score(y_val, refined)
        print(f"  fraction={frac:.2f}  accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_fraction = frac

    print(f"\nBest prefix fraction: {best_fraction}  (val accuracy: {best_acc:.4f})")

    # ----------------------------------------------------------
    # Full validation report with chosen fraction
    # ----------------------------------------------------------
    print("\nFull validation report:")
    best_refiner = PrefixZeroVsTwoRefiner(fraction=best_fraction)
    best_refiner.fit(recs_train, y_train)

    refined_pred = base_pred.copy()
    candidate_idx = [i for i, p in enumerate(base_pred) if p in [0, 2]]
    if candidate_idx:
        refined_02 = best_refiner.predict([recs_val[i] for i in candidate_idx])
        for j, label in zip(candidate_idx, refined_02):
            refined_pred[j] = label

    print(classification_report(y_val, refined_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, refined_pred))

    # ----------------------------------------------------------
    # Train final merged model on all data and save
    # ----------------------------------------------------------
    print(f"\nTraining final PathClassifier (fraction={best_fraction}) on all data...")
    final_model = PathClassifier(prefix_fraction=best_fraction, n_segments=8)
    final_model.fit(recordings, y)

    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(final_model, f)
    print(f"Saved: {OUTPUT_MODEL}")
    print("\nInference example:")
    print("  with open('path_model.pkl', 'rb') as f: model = pickle.load(f)")
    print("  path  = model.predict_one(recording)")


if __name__ == "__main__":
    main()
