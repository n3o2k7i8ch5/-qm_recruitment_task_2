import pandas as pd

PREDICTION_RESULTS_FILE = 'prediction_results.txt'
MODEL_PICKLE_FILE = 'model.pkl'
TRAIN_DATA_FILE = 'data/train.csv'
TEST_DATA_FILE = 'data/hidden_test.csv'


def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    # Data analysis revealed that only the three columns below turned out to provide more
    # information than noise. It might be the case that other attributes also slightly
    # contribute to enhancing the score, but only within the margin of error.
    return X.iloc[:, [6, 7, 8]]
