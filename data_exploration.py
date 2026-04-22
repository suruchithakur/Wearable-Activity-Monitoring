import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from mhealth_activity import Recording, Activity, WatchLocation, Path

def explore_dataset(data_dir):
    print(f"Exploring dataset in: {data_dir}")
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    print(f"Found {len(pkl_files)} .pkl files.")

    activities_counts = Counter()
    watch_loc_counts = Counter()
    path_idx_counts = Counter()
    durations = []
    sensor_counts = Counter()
    
    # Analyze a subset if there are too many, but 396 is manageable
    for i, file_path in enumerate(pkl_files):
        if i % 50 == 0:
            print(f"Processing file {i}/{len(pkl_files)}...")
            
        try:
            recording = Recording(file_path)
            
            # Labels
            labels = recording.labels
            if 'activities' in labels:
                for act in labels['activities']:
                    activities_counts[Activity(act).name] += 1
            if 'watch_loc' in labels:
                watch_loc_counts[WatchLocation(labels['watch_loc']).name] += 1
            if 'path_idx' in labels:
                path_idx_counts[f"PATH_{labels['path_idx']}"] += 1
                
            # Durations (using 'ax' as reference if available)
            if 'ax' in recording.data:
                durations.append(recording.data['ax'].total_time)
            
            # Sensors
            for sensor in recording.data.keys():
                sensor_counts[sensor] += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\n--- Statistics ---")
    print(f"Total Recordings: {len(pkl_files)}")
    print(f"Average Duration: {np.mean(durations):.2f}s (Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s)")
    
    print("\nActivities Distribution:")
    for act, count in activities_counts.items():
        print(f"  {act}: {count}")
        
    print("\nWatch Location Distribution:")
    for loc, count in watch_loc_counts.items():
        print(f"  {loc}: {count}")
        
    print("\nPath Index Distribution:")
    for p, count in sorted(path_idx_counts.items()):
        print(f"  {p}: {count}")

    print("\nCommon Sensors Availability:")
    for sensor, count in sensor_counts.most_common(10):
        print(f"  {sensor}: {count}/{len(pkl_files)}")

    # Visualizations
    plot_distributions(activities_counts, watch_loc_counts, path_idx_counts)
    
    # Example Individual Plot
    if pkl_files:
        example_recording = Recording(pkl_files[0])
        plot_example(example_recording)

def plot_distributions(activities, watch_locs, paths):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Activities
    axes[0].bar(activities.keys(), activities.values(), color='skyblue')
    axes[0].set_title('Activities Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Watch Locations
    axes[1].bar(watch_locs.keys(), watch_locs.values(), color='salmon')
    axes[1].set_title('Watch Location Distribution')
    
    # Paths
    axes[2].bar(paths.keys(), paths.values(), color='lightgreen')
    axes[2].set_title('Path Distribution')
    
    plt.tight_layout()
    plt.savefig('dataset_distributions.png')
    print("\nSaved distribution plot to 'dataset_distributions.png'")

def plot_example(recording):
    # Plot accelerometer magnitude and GPS if available
    if 'ax' in recording.data and 'ay' in recording.data and 'az' in recording.data:
        ax = recording.data['ax'].values
        ay = recording.data['ay'].values
        az = recording.data['az'].values
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        plt.figure(figsize=(12, 4))
        plt.plot(recording.data['ax'].timestamps, mag, label='Acc Magnitude', color='black', alpha=0.7)
        plt.title(f'Accelerometer Magnitude for {recording.filename}')
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude (g)')
        plt.grid(True)
        plt.savefig('example_acc_mag.png')
        print(f"Saved example Acc Mag plot to 'example_acc_mag.png'")

    if 'latitude' in recording.data and 'longitude' in recording.data:
        plt.figure(figsize=(8, 8))
        plt.scatter(recording.data['longitude'].values, recording.data['latitude'].values, s=1, c='blue')
        plt.title(f'GPS Trajectory for {recording.filename}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axis('equal')
        plt.grid(True)
        plt.savefig('example_gps_path.png')
        print(f"Saved example GPS plot to 'example_gps_path.png'")

if __name__ == "__main__":
    train_dir = "data/train"
    if os.path.exists(train_dir):
        explore_dataset(train_dir)
    else:
        print(f"Error: Directory {train_dir} not found. Please run from the '26-mham-ex3' folder.")
