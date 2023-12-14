import os
import pickle

import pandas as pd

from utils import MODEL_PICKLE_FILE, TEST_DATA_FILE, PREDICTION_RESULTS_FILE, \
    preprocess_data

if __name__ == '__main__':
    if not os.path.exists(MODEL_PICKLE_FILE):
        print('Failed: No model found. To create a model call `python train.py`.')
        exit()

    with open(MODEL_PICKLE_FILE, 'rb') as file:
        # Load the trained XGBRegressor model.
        model = pickle.load(file)

    X = preprocess_data(pd.read_csv(TEST_DATA_FILE))

    y_pred = model.predict(X)

    with open(PREDICTION_RESULTS_FILE, 'w') as file:
        # Save predictions to file.
        file.write("\n".join([str(val) for val in y_pred]))

    print(f'Used model from "{MODEL_PICKLE_FILE}" and saved predictions of "{TEST_DATA_FILE}" to "{PREDICTION_RESULTS_FILE}".')
