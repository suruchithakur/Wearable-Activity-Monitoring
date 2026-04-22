import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train():
    print("Loading features...")
    df = pd.read_csv('train_features.csv')
    
    # Define features to use for classification
    # We exclude filename, activities (list), and the targets
    exclude_cols = ['filename', 'activities', 'activity_id', 'watch_loc', 'path_idx', 'step_count']
    X_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Using {len(X_cols)} features for training.")
    
    # Handle NaNs (missing sensors)
    X = df[X_cols].fillna(-1)
    
    # 1. Activity Model
    print("\nTraining Activity Model...")
    y_act = df['activity_id']
    X_train, X_test, y_train, y_test = train_test_split(X, y_act, test_size=0.2, random_state=42, stratify=y_act)
    
    act_model = RandomForestClassifier(n_estimators=100, random_state=42)
    act_model.fit(X_train, y_train)
    preds = act_model.predict(X_test)
    print("Activity Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    
    # 2. Watch Location Model
    print("\nTraining Watch Location Model...")
    y_loc = df['watch_loc']
    X_train, X_test, y_train, y_test = train_test_split(X, y_loc, test_size=0.2, random_state=42, stratify=y_loc)
    
    loc_model = RandomForestClassifier(n_estimators=100, random_state=42)
    loc_model.fit(X_train, y_train)
    preds = loc_model.predict(X_test)
    print("Watch Location Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    
    # 3. Path Index Model
    print("\nTraining Path Index Model...")
    y_path = df['path_idx']
    X_train, X_test, y_train, y_test = train_test_split(X, y_path, test_size=0.2, random_state=42, stratify=y_path)
    
    path_model = RandomForestClassifier(n_estimators=100, random_state=42)
    path_model.fit(X_train, y_train)
    preds = path_model.predict(X_test)
    print("Path Index Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(act_model, 'models/activity_model.pkl')
    joblib.dump(loc_model, 'models/loc_model.pkl')
    joblib.dump(path_model, 'models/path_model.pkl')
    joblib.dump(X_cols, 'models/feature_cols.pkl')
    print("\nModels saved to models/ directory.")

if __name__ == "__main__":
    train()
