import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import find_peaks

def extract_features(recording):
    # recording is a Recording object from mhealth_activity
    features = {}
    
    # We focus on Accelerometer (ax, ay, az) as it is most consistently available
    if 'ax' in recording.data and 'ay' in recording.data and 'az' in recording.data:
        ax = recording.data['ax'].values
        ay = recording.data['ay'].values
        az = recording.data['az'].values
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        # Time domain features
        for name, signal in [('ax', ax), ('ay', ay), ('az', az), ('mag', mag)]:
            features[f'{name}_mean'] = np.mean(signal)
            features[f'{name}_std'] = np.std(signal)
            features[f'{name}_max'] = np.max(signal)
            features[f'{name}_min'] = np.min(signal)
            features[f'{name}_rms'] = np.sqrt(np.mean(signal**2))
            
            # Zero-crossing rate (on detrended signal)
            detrended = signal - np.mean(signal)
            features[f'{name}_zcr'] = np.mean(np.diff(np.sign(detrended)) != 0)
            
        # Frequency domain features (on magnitude)
        n = len(mag)
        f_mag = np.abs(fft(mag - np.mean(mag))[:n//2])
        freqs = np.linspace(0, recording.data['ax'].samplerate / 2, n // 2)
        
        # Energy in bands
        # Walking: 0.5 - 3 Hz
        # Running: 3 - 8 Hz
        walking_band = (freqs >= 0.5) & (freqs <= 3)
        running_band = (freqs > 3) & (freqs <= 8)
        
        features['mag_energy_walking'] = np.sum(f_mag[walking_band]**2) if any(walking_band) else 0
        features['mag_energy_running'] = np.sum(f_mag[running_band]**2) if any(running_band) else 0
        
        # Dominant frequency
        if len(f_mag) > 0:
            features['mag_dom_freq'] = freqs[np.argmax(f_mag)]
        else:
            features['mag_dom_freq'] = 0
            
    # Other sensors (Gyroscope)
    if 'gx' in recording.data:
        for axis in ['gx', 'gy', 'gz']:
            val = recording.data[axis].values
            features[f'{axis}_std'] = np.std(val)
            features[f'{axis}_mean'] = np.mean(val)

    # Smartphone sensors (excluding forbidden keys: longitude, bearing, speed, phone_steps)
    for key in ['altitude']: # Only altitude is kept from the previous 'smartphone' list
        if key in recording.data:
            features[key] = np.mean(recording.data[key].values)
        else:
            features[key] = np.nan
            
    return features

if __name__ == "__main__":
    import os
    import glob
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
    df_final = pd.merge(metadata, df_features, on='filename')
    df_final.to_csv('train_features.csv', index=False)
    print("Done! Saved to train_features.csv")
