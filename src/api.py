from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import sys
sys.path.append('/root/ml_process_feb23/')
import src.util as utils
import src.data_pipeline as pipeline
import src.preprocessing as preprocessing
import src.modelling as modelling

# Loading configuration file
config = utils.load_config()

# Loading production model
model_data = utils.pkl_load(config["production_model_path"])

# Define input data
class api_data(BaseModel):

    owner: str
    selfemp: str
    reports: int
    age: int
    dependents: int
    months: int
    majorcards: int
    active: int
    income: float
    share: float
    expenditure: float

app = FastAPI()

# Landing page
@app.get("/")
def home():
    return "Activating FastAPI!"

# Prediction page
@app.post("/predict/")
def predict(data: api_data):    
    # Convert data to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)  # type: ignore

    # Convert dtype
    data = pd.concat(
        [
            data[config["predictors_api"][0:2]],
            data[config["predictors_api"][2:8]].astype(int),
            data[config["predictors_api"][8:]].astype(float)
        ],
        axis = 1
    )

    # Data transformation
    data = preprocessing.cols_transform(data, config["cols_to_log"])

    # Label encoding
    data = preprocessing.label_encoding(data, config["predictors_cat"])

    # Data binning
    data = preprocessing.binning(data, 'age', config["bins_age"], config["labels_age"])
    data = preprocessing.binning(data, 'reports', config["bins_reports"], config["labels_reports"])
    data = preprocessing.binning(data, 'dependents', config["bins_dependents"], config["labels_dependents"])
    data = preprocessing.binning(data, 'active', config["bins_active"], config["labels_active"])

    # Include the binned features for fitting the model
    data = modelling.get_binned_features(data, config["original_cols"])

    # Value division
    data = preprocessing.division(data, 'income_log')

    # Predict data
    y_pred = model_data.predict(data)

    if y_pred[0] == 0:
        y_pred = "Credit card application is not approved."
    else:
        y_pred = "Your credit card application is approved!"
    return {"result" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)

