import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from mhealth_activity import Recording
from train_path_model import (
    PathClassifier,
    TwoStagePathClassifier,
    PrefixZeroVsTwoRefiner,
    extract_path_features,
)


def get_feature_cols(df):
    """Get feature column names (exclude metadata and target columns)."""
    exclude_cols = [
        'filename', 'activities', 'activity_id',
        'watch_loc', 'path_idx', 'step_count',
        'has_standing', 'has_walking', 'has_running', 'has_cycling',
        'phone_steps'
    ]
    return [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]


def evaluate_cv(model, X, y, task_name, cv=5):
    """Evaluate model with stratified cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"  CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean()


def make_rf(n_estimators=500, class_weight=None):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42
    )


def train():
    print("=" * 60)
    print("TRAINING PIPELINE - Multi-Label Activity + Classification")
    print("=" * 60)

    print("\nLoading features...")
    df = pd.read_csv('train_features.csv')
    print(f"Dataset: {len(df)} samples")

    X_cols = get_feature_cols(df)
    print(f"Using {len(X_cols)} features")

    # Impute NaN / inf with median
    X_raw = df[X_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_cols)
    X = X.fillna(0)

    os.makedirs('models', exist_ok=True)

    # ================================================================
    # 1. MULTI-LABEL ACTIVITY CLASSIFICATION (4 binary classifiers)
    # ================================================================
    print("\n" + "=" * 60)
    print("1. ACTIVITY RECOGNITION (Multi-Label, 4 binary classifiers)")
    print("=" * 60)

    for act_name in ['has_standing', 'has_walking', 'has_running', 'has_cycling']:
        print(f"\n  --- {act_name} ---")
        y = df[act_name].astype(int)
        pos = y.sum(); neg = len(y) - pos
        print(f"  Positive: {pos}, Negative: {neg}")

        # Balanced class weight handles imbalance automatically
        model = make_rf(n_estimators=500, class_weight='balanced')
        evaluate_cv(model, X, y, act_name)
        model.fit(X, y)
        joblib.dump(model, f'models/{act_name}_model.pkl')
        print(f"  Saved models/{act_name}_model.pkl")

    # ================================================================
    # 2. WATCH LOCATION CLASSIFICATION
    # ================================================================
    print("\n" + "=" * 60)
    print("2. WATCH LOCATION")
    print("=" * 60)
    y_loc = df['watch_loc'].astype(int)
    print(f"  Distribution: {dict(y_loc.value_counts().sort_index())}")

    loc_model = make_rf(n_estimators=500)
    evaluate_cv(loc_model, X, y_loc, 'Watch Location')
    loc_model.fit(X, y_loc)
    joblib.dump(loc_model, 'models/loc_model.pkl')
    print("  Saved models/loc_model.pkl")

    # ================================================================
    # 3. PATH INDEX CLASSIFICATION (TwoStage + Prefix Refiner)
    # ================================================================
    print("\n" + "=" * 60)
    print("3. PATH INDEX (TwoStagePathClassifier + PrefixZeroVsTwoRefiner)")
    print("=" * 60)

    TRAIN_DIR = "data/train"
    FRACTIONS_TO_TRY = [0.15, 0.20, 0.25, 0.30, 0.35]

    print("  Loading recordings...")
    recordings, path_labels = [], []
    for filename in sorted(os.listdir(TRAIN_DIR)):
        if not filename.endswith(".pkl"):
            continue
        rec = Recording(os.path.join(TRAIN_DIR, filename))
        if rec.labels is None:
            continue
        recordings.append(rec)
        path_labels.append(rec.labels["path_idx"])

    y_path = np.array(path_labels)
    print(f"  Loaded {len(recordings)} recordings")
    print(f"  Distribution: {dict(pd.Series(y_path).value_counts().sort_index())}")

    print("  Extracting path features...")
    X_path = pd.DataFrame([extract_path_features(r, n_segments=8) for r in recordings])

    indices = np.arange(len(recordings))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_path
    )
    X_train_p, X_val_p = X_path.iloc[train_idx], X_path.iloc[val_idx]
    y_train_p, y_val_p = y_path[train_idx], y_path[val_idx]
    recs_train = [recordings[i] for i in train_idx]
    recs_val = [recordings[i] for i in val_idx]

    print("  Fitting main two-stage model on train split...")
    main_model = TwoStagePathClassifier()
    main_model.fit(X_train_p, y_train_p)
    base_pred = main_model.predict(X_val_p)
    print(f"  Base accuracy (no refiner): {accuracy_score(y_val_p, base_pred):.4f}")

    print("  Searching for best prefix fraction...")
    best_fraction, best_acc = None, -1.0
    for frac in FRACTIONS_TO_TRY:
        refiner = PrefixZeroVsTwoRefiner(fraction=frac)
        refiner.fit(recs_train, y_train_p)
        refined = base_pred.copy()
        candidate_idx = [i for i, p in enumerate(base_pred) if p in [0, 2]]
        if candidate_idx:
            refined_02 = refiner.predict([recs_val[i] for i in candidate_idx])
            for j, label in zip(candidate_idx, refined_02):
                refined[j] = label
        acc = accuracy_score(y_val_p, refined)
        print(f"    fraction={frac:.2f}  accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_fraction = frac

    print(f"  Best prefix fraction: {best_fraction}  (val accuracy: {best_acc:.4f})")

    print("  Full validation report:")
    best_refiner = PrefixZeroVsTwoRefiner(fraction=best_fraction)
    best_refiner.fit(recs_train, y_train_p)
    refined_pred = base_pred.copy()
    candidate_idx = [i for i, p in enumerate(base_pred) if p in [0, 2]]
    if candidate_idx:
        refined_02 = best_refiner.predict([recs_val[i] for i in candidate_idx])
        for j, label in zip(candidate_idx, refined_02):
            refined_pred[j] = label
    print(classification_report(y_val_p, refined_pred))
    print("  Confusion matrix:")
    print(confusion_matrix(y_val_p, refined_pred))

    print(f"  Training final PathClassifier (fraction={best_fraction}) on all data...")
    path_model = PathClassifier(prefix_fraction=best_fraction, n_segments=8)
    path_model.fit(recordings, y_path)
    with open('models/path_model.pkl', 'wb') as f:
        pickle.dump(path_model, f)
    print("  Saved models/path_model.pkl")

    # ================================================================
    # SAVE METADATA
    # ================================================================
    joblib.dump(X_cols, 'models/feature_cols.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')

    print("\n" + "=" * 60)
    print("ALL MODELS SAVED")
    print("=" * 60)


if __name__ == "__main__":
    train()
