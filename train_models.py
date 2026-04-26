import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


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
    # 3. PATH INDEX CLASSIFICATION
    # ================================================================
    print("\n" + "=" * 60)
    print("3. PATH INDEX")
    print("=" * 60)
    y_path = df['path_idx'].astype(int)
    print(f"  Distribution: {dict(y_path.value_counts().sort_index())}")

    path_model = make_rf(n_estimators=500)
    evaluate_cv(path_model, X, y_path, 'Path Index')
    path_model.fit(X, y_path)
    joblib.dump(path_model, 'models/path_model.pkl')
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
