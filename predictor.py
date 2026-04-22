import os
import joblib
import numpy as np
import pandas as pd
from feature_extractor import extract_features
from step_counter import count_steps
from mhealth_activity import Activity, WatchLocation

class CompetitionPredictor:
    def __init__(self, models_dir='models'):
        self.act_model = joblib.load(os.path.join(models_dir, 'activity_model.pkl'))
        self.loc_model = joblib.load(os.path.join(models_dir, 'loc_model.pkl'))
        self.path_model = joblib.load(os.path.join(models_dir, 'path_model.pkl'))
        self.feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))

    def predict(self, recording):
        # 1. Extract Features
        feats = extract_features(recording)
        
        # 2. Prepare Feature Vector for classification models
        x_df = pd.DataFrame([feats])
        X = x_df[self.feature_cols].fillna(-1)
        
        # 3. Classify Category Targets
        act_id = self.act_model.predict(X)[0]
        watch_loc = self.loc_model.predict(X)[0]
        path_idx = self.path_model.predict(X)[0]
        
        # 4. Step Count (with activity refinement)
        if act_id in [1, 2]: # WALKING or RUNNING
            steps = count_steps(recording)
        else:
            steps = 0
            
        # 5. Output in target format
        # Activity ID mapping to individual booleans
        return {
            'watch_loc': int(watch_loc),
            'path_idx': int(path_idx),
            'standing': bool(act_id == 0),
            'walking': bool(act_id == 1),
            'running': bool(act_id == 2),
            'cycling': bool(act_id == 3),
            'step_count': int(steps)
        }

if __name__ == "__main__":
    from mhealth_activity import Recording
    predictor = CompetitionPredictor()
    
    # Test on one file
    rec = Recording('data/train/train_trace_015.pkl')
    pred = predictor.predict(rec)
    print("Example Prediction for train_trace_015.pkl:")
    print(pred)
