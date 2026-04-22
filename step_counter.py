import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

def count_steps(recording):
    # recording is a Recording object
    if 'ax' not in recording.data:
        return 0
    
    # We use Accelerometer Magnitude
    ax = recording.data['ax'].values
    ay = recording.data['ay'].values
    az = recording.data['az'].values
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    
    fs = recording.data['ax'].samplerate
    
    # 1. Bandpass filter (0.5 to 5 Hz is the typical range for human gait)
    # If the recording is too short, skip filtering
    if len(mag) > 15:
        try:
            nyq = 0.5 * fs
            low = 0.5 / nyq
            high = 5.0 / nyq
            b, a = butter(4, [low, high], btype='band')
            filtered_mag = filtfilt(b, a, mag)
        except Exception:
            filtered_mag = mag - np.mean(mag)
    else:
        filtered_mag = mag - np.mean(mag)
        
    # 2. Peak Detection
    # height: threshold to detect a step (empirically around 0.1-0.3 for filtered mag)
    # distance: minimum number of samples between steps (~0.3s)
    peaks, _ = find_peaks(filtered_mag, height=0.1, distance=int(0.3 * fs))
    
    return len(peaks)

if __name__ == "__main__":
    import os
    import pandas as pd
    from mhealth_activity import Recording
    
    metadata = pd.read_csv('train_metadata.csv')
    # Filter rows where step_count is valid (>0)
    valid_steps = metadata[metadata['step_count'] > 0].head(20)
    
    print("Testing Step counter on training data...")
    errors = []
    for i, row in valid_steps.iterrows():
        f_path = os.path.join('data/train', row['filename'])
        rec = Recording(f_path)
        pred = count_steps(rec)
        actual = int(row['step_count'])
        error = abs(pred - actual) / actual
        print(f"File: {row['filename']}, Pred: {pred}, Actual: {actual}, Error: {error:.2%}")
        errors.append(error)
        
    print(f"\nMean Relative Error: {np.mean(errors):.2%}")
