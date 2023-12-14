import pandas as pd
import pickle
from xgboost import XGBRegressor

from utils import MODEL_PICKLE_FILE, TRAIN_DATA_FILE, preprocess_data

if __name__ == "__main__":
    # XGBoost Regressor with the provided params turned out to be the best regression
    # model based on the second part of the `data_and_model_analysis.ipynb` jupyter
    # notebook.
    model = XGBRegressor(n_estimators=1000, n_jobs=10)

    train = pd.read_csv(TRAIN_DATA_FILE)
    y = train.pop('target')
    X = preprocess_data(train)

    model.fit(X, y)

    with open(MODEL_PICKLE_FILE, 'wb') as file:
        # Dump the trained model to enable calling `predict.py` as a standalone script.
        pickle.dump(model, file)

    print(f'Model trained and saved to "{MODEL_PICKLE_FILE}".')
