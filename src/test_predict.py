import pandas as pd
import sys
PATH = "/home/er_bim/ML-Process-Final-Project-Feb23-Bimo/"
sys.path.append(PATH)
import src.util as utils
import src.preprocessing as preprocessing
import src.modelling as modelling

if __name__ == "__main__":

    # Loading configuration file
    config = utils.load_config()

    # Loading production model
    model_data = utils.pkl_load(config["production_model_path"])

    # Define the predictors
    data = {
        "reports": [2],
        "age": [20],
        "income": [30],
        "share": [.2],
        "expenditure": [700],
        "owner": ["no"],
        "selfemp": ["no"],
        "dependents": [2],
        "months": [30],
        "majorcards": [1],
        "active": [4]
    }

    # Convert to DataFrame
    predictors = pd.DataFrame(data)

    # Data transformation
    predictors = preprocessing.cols_transform(predictors, config["cols_to_log"])

    # Label encoding
    predictors = preprocessing.label_encoding(predictors, config["predictors_cat"])

    # Data binning
    predictors = preprocessing.binning(predictors, 'age', config["bins_age"], config["labels_age"])
    predictors = preprocessing.binning(predictors, 'reports', config["bins_reports"], config["labels_reports"])
    predictors = preprocessing.binning(predictors, 'dependents', config["bins_dependents"], config["labels_dependents"])
    predictors = preprocessing.binning(predictors, 'active', config["bins_active"], config["labels_active"])

    # Include the binned features for fitting the model
    predictors = modelling.get_binned_features(predictors, config["original_cols"])

    # Value division
    predictors = preprocessing.division(predictors, 'income_log')

    # Check the result
    print(model_data.predict(predictors))