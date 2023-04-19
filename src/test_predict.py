import pandas as pd
import sys
sys.path.append('/root/ml_process_feb23/')
import src.util as utils

if __name__ == "__main__":

    # Loading configuration file
    config = utils.load_config()

    # Loading production model
    model_data = utils.pkl_load(config["production_model_path"])

    # Define the prodictors
    data = {
        "reports": [],
        "age": [],
        "income": [],
        "share": [],
        "expenditure": [],
        "owner": [],
        "selfemp": [],
        "dependents": [],
        "months": [],
        "majorcards": [],
        "active": []
    }

    # Convert to DataFrame
    predictors = pd.DataFarame(data)

    # Check the result
    print(model_data.predict(predictors))