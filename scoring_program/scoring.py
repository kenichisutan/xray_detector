# The scoring program compute scores from:
# - The ground truth
# - The predictions made by the candidate model

# Imports
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Path
input_dir = '/app/input'    # Input from ingestion program
output_dir = '/app/output/'  # To write the scores
reference_dir = os.path.join(input_dir, 'ref')  # Ground truth data
prediction_dir = os.path.join(input_dir, 'res')  # Prediction made by the model
score_file = os.path.join(output_dir, 'scores.json')          # Scores
html_file = os.path.join(output_dir, 'detailed_results.html')  # Detailed feedback


def write_file(file, content):
    """ Write content in file.
    """
    with open(file, 'a', encoding="utf-8") as f:
        f.write(content)


def get_data():
    """ Get ground truth (y_test) and predictions (y_pred).
    """
    y_test = pd.read_csv(os.path.join(reference_dir, 'test_labels.csv'), header=None)
    y_test = np.array(y_test)
    y_test = y_test.ravel()
    y_pred = np.genfromtxt(os.path.join(prediction_dir, 'predict'))
    return y_test, y_pred


def main():
    """ The scoring program.
    """
    print('Scoring program.')
    scores = {}
    # Read data
    print('Reading prediction')
    y_test, y_pred = get_data()
    print(y_test.shape, y_pred.shape)
    # Compute scores
    mse = mean_squared_error(y_test, y_pred)
    print('MSE: {}'.format(mse))
    scores['MSE'] = mse
    r2 = r2_score(y_test, y_pred)
    print('R2: {}'.format(r2))
    scores['R2'] = r2
    # Get duration
    with open(os.path.join(prediction_dir, 'metadata.json')) as f:
        duration = json.load(f).get('duration', -1)
    scores['duration'] = duration
    # Write scores
    print('Scoring program finished. Writing scores.')
    print(scores)
    write_file(score_file, json.dumps(scores))


if __name__ == '__main__':
    main()
