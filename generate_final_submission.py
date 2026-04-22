import os
import re
import pandas as pd
from mhealth_activity import Recording
from predictor import CompetitionPredictor

def generate_submission(test_dir='data/test', output_file='submission.csv'):
    predictor = CompetitionPredictor(models_dir='models')
    
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.pkl')])
    print(f"Processing {len(test_files)} test files...")
    
    submission_data = []
    
    for i, filename in enumerate(test_files):
        if i % 20 == 0:
            print(f"Progress: {i}/{len(test_files)}")
            
        # Extract ID
        match = re.search(r'(\d{3})\.pkl$', filename)
        if match:
            file_id = int(match.group(1))
        else:
            print(f"Warning: Could not extract ID from {filename}")
            continue
            
        # Load and predict
        try:
            f_path = os.path.join(test_dir, filename)
            rec = Recording(f_path)
            preds = predictor.predict(rec)
            
            # Combine ID with predictions
            row = {'Id': file_id}
            row.update(preds)
            submission_data.append(row)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # Add a 'safe' default row to avoid missing entries
            submission_data.append({
                'Id': file_id,
                'watch_loc': 0,
                'path_idx': 0,
                'standing': False,
                'walking': False,
                'running': False,
                'cycling': False,
                'step_count': 0
            })

    # Create DataFrame and Save
    df = pd.DataFrame(submission_data, columns=['Id', 'watch_loc', 'path_idx', 'standing', 'walking', 'running', 'cycling', 'step_count'])
    df.to_csv(output_file, index=False)
    print(f"\nDone! Submission saved to {output_file}")

if __name__ == "__main__":
    generate_submission()
