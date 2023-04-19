from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import sys
sys.path.append('/root/ml_process_feb23/')
import src.util as utils
import src.data_pipeline as pipeline
import src.preprocessing as preprocessing

# Loading configuration file
config = utils.load_config()

# Loading production model
model_data = utils.pkl_load(config["production_model_path"])

# Define input data
class InputData(BaseModel):

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
def predict(data: InputData):    
    # Convert data to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)  # type: ignore
    data.columns = config["predictors_api"]

    # Convert dtype
    data = pd.concat(
        [
            data[config["predictors_api"][0:2]],
            data[config["predictors_api"][2:8]].astype(int),
            data[config["predictors_api"][8:]].astype(float)
        ],
        axis = 1
    )

    # Check range data
    try:
        pipeline.check_data(data, config, True)  # type: ignore
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}

    # Predict data
    y_pred = model_data.predict(data)

    if y_pred[0] == 0:
        y_pred = "Credit card application is not approved."
    else:
        y_pred = "Your credit card application is approved!"
    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)

