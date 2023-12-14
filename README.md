# QM recruitment task 2

This repository contains the solution to the second recruitment task from QM.

## Installation

Use the package manager pip to install the required dependencies.
Place the files `hidden_test.csv` and `train.csv` in the `data/` path.
These files can be obtained from QM as a recruitment task.  

```bash
pip install -r requirements.txt
```

## Usage
### Data analysis
Data analysis and ML regression models comparison was performed and saved in the `data_and_model_analysis.ipynb` jupyter notebook.

### Model training
In order to train a `XGBRegressor(n_estimators=1000)` model execute
```bash
python train.py
```
This will create, train and save a regression model based on the `train.csv` data.

### Test data prediction
In order create target predictions for the `hidden_test.csv` data, execute
```bash
python predict.py
```
This will load and use the prediction model and generate predictions.
