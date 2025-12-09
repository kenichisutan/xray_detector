import json
import os
import sys
import time
import numpy as np
import pandas as pd

# Paths
input_dir = '/app/input_data/'  # Input data
output_dir = '/app/output/'    # For the predictions
program_dir = '/app/program'
submission_dir = '/app/ingested_program'  # The code submitted
sys.path.append(output_dir)
sys.path.append(program_dir)
sys.path.append(submission_dir)


def get_data():
    """ Get X_train, y_train, X_adapt and X_test from csv files
    """
    # Read data
    X_train = pd.read_csv(os.path.join(input_dir, 'train.csv'), header=None)
    y_train = pd.read_csv(os.path.join(input_dir, 'train_labels.csv'), header=None)
    X_adapt = pd.read_csv(os.path.join(input_dir, 'train_DA.csv'), header=None)
    X_test = pd.read_csv(os.path.join(input_dir, 'test.csv'), header=None)
    # Convert to numpy arrays
    X_train, y_train, X_adapt, X_test = np.array(X_train), np.array(y_train), np.array(X_adapt), np.array(X_test)
    y_train = y_train.ravel()
    return X_train, y_train, X_adapt, X_test


def print_bar():
    """ Display a bar ('----------')
    """
    print('-' * 10)


def main():
    """ The ingestion program.
    """
    print_bar()
    print('Ingestion program.')
    from model import Model  # The model submitted by the participant
    start = time.time()
    # Read data
    print('Reading data')
    X_train, y_train, X_adapt, X_test = get_data()
    # Initialize model
    print('Initializing the model')
    m = Model()
    # Train model
    print('Training the model')
    m.fit(X_train, y_train, X_adapt)
    # Make predictions
    print('Making predictions')
    y_pred = m.predict(X_test)
    # Save predictions
    np.savetxt(os.path.join(output_dir, 'predict'), y_pred)
    duration = time.time() - start
    print(f'Time elapsed so far: {duration}')
    # End
    duration = time.time() - start
    print(f'Completed. Total duration: {duration}')
    with open(os.path.join(output_dir, 'metadata.json'), 'w+') as f:
        json.dump({'duration': duration}, f)
    print('Ingestion program finished. Moving on to scoring')
    print_bar()


if __name__ == '__main__':
    main()
