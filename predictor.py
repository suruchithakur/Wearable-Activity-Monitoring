import os
import joblib
import numpy as np
import pandas as pd
from feature_extractor import extract_features
from step_counter import count_steps


class CompetitionPredictor:
    def __init__(self, models_dir='models'):
        # Multi-label activity models
        self.act_standing_model = joblib.load(os.path.join(models_dir, 'has_standing_model.pkl'))
        self.act_walking_model = joblib.load(os.path.join(models_dir, 'has_walking_model.pkl'))
        self.act_running_model = joblib.load(os.path.join(models_dir, 'has_running_model.pkl'))
        self.act_cycling_model = joblib.load(os.path.join(models_dir, 'has_cycling_model.pkl'))

        # Classification models
        self.loc_model = joblib.load(os.path.join(models_dir, 'loc_model.pkl'))
        self.path_model = joblib.load(os.path.join(models_dir, 'path_model.pkl'))

        # Feature metadata
        self.feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
        self.imputer = joblib.load(os.path.join(models_dir, 'imputer.pkl'))

    def predict(self, recording):
        # 1. Extract Features
        feats = extract_features(recording)

        # 2. Prepare Feature Vector
        x_df = pd.DataFrame([feats])

        # Ensure all expected columns exist (fill missing with NaN)
        for col in self.feature_cols:
            if col not in x_df.columns:
                x_df[col] = np.nan

        X_raw = x_df[self.feature_cols]
        X = pd.DataFrame(self.imputer.transform(X_raw), columns=self.feature_cols)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 3. Multi-label Activity Prediction
        standing = bool(self.act_standing_model.predict(X)[0])
        walking = bool(self.act_walking_model.predict(X)[0])
        running = bool(self.act_running_model.predict(X)[0])
        cycling = bool(self.act_cycling_model.predict(X)[0])

        # Ensure at least one activity is predicted
        if not any([standing, walking, running, cycling]):
            # Fall back to walking as most common activity
            walking = True

        # 4. Watch Location
        watch_loc = int(self.loc_model.predict(X)[0])

        # 5. Path Index
        path_idx = int(self.path_model.predict(X)[0])

        # 6. Step Count
        # Count steps if walking or running is predicted (or both)
        if walking or running:
            steps = count_steps(recording, watch_loc=watch_loc)
        else:
            steps = 0

        # If cycling only, ensure 0 steps
        if cycling and not walking and not running:
            steps = 0

        # If standing only, ensure 0 steps
        if standing and not walking and not running:
            steps = 0

        return {
            'watch_loc': watch_loc,
            'path_idx': path_idx,
            'standing': standing,
            'walking': walking,
            'running': running,
            'cycling': cycling,
            'step_count': int(steps)
        }


if __name__ == "__main__":
    from mhealth_activity import Recording
    predictor = CompetitionPredictor()

    # Test on one file
    rec = Recording('data/train/train_trace_015.pkl')
    pred = predictor.predict(rec)
    print("Example Prediction for train_trace_015.pkl:")
    print(f"  Labels: {rec.labels}")
    print(f"  Prediction: {pred}")
