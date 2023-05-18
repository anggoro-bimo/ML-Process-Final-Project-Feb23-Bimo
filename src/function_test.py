import pandas as pd
import numpy as np
import sys
PATH = "/home/er_bim/ML-Process-Final-Project-Feb23-Bimo/"
sys.path.append(PATH)
import src.util as utils
import preprocessing



def test_cols_transform():
    # Arrange
    mock_data = {
        "income" : [100],
        "expenditure" : [10]
    }
    mock_data = pd.DataFrame(mock_data)
    
    expected_data = {
        "income_log" : [np.log10(100+1)],
        "expenditure_log" : [np.log10(10+1)]
    }
    expected_data = pd.DataFrame(expected_data)
    
    # Act
    processed_data = preprocessing.cols_transform(mock_data, ["income", "expenditure"])
    
    # Assert
    assert processed_data.equals(expected_data)
    
def test_label_encoding():
    # Arrange
    mock_data = {
        "owner" : ["no", "yes"],
        "selfemp" : ["no", "yes"],
        "card" : ["no", "yes"],
     }
    mock_data = pd.DataFrame(mock_data)

    expected_data = {
            "owner" : [0, 1],
            "selfemp" : [0, 1],
            "card" : [0, 1],
     }
    expected_data = pd.DataFrame(expected_data)
     
    # Act
    processed_data = preprocessing.label_encoding(mock_data, ["owner", "selfemp", "card"])

    # Assert
    assert processed_data.equals(expected_data)
    
def test_binning():
    # Arrange
    config = utils.load_config()
    mock_data = {"age" : [18, 30, 51, 65, 77]}
    mock_data = pd.DataFrame(mock_data)
    
    expected_data = {
        "age" : [18, 30, 51, 65, 77],
        "age_bin" : [1, 2, 3, 4, 5]
    }
    expected_data = pd.DataFrame(expected_data)
    
    # Act
    processed_data = preprocessing.binning(mock_data, "age", config["bins_age"], config["labels_age"])
    
    # Assert
    assert processed_data.equals(expected_data)

def test_division():
    # Arrange
    mock_data = {"income_log" : [12]}
    mock_data = pd.DataFrame(mock_data)
    
    expected_data = {"income_log" : [1.0]}
    expected_data = pd.DataFrame(expected_data)
    
    # Act
    processed_data = preprocessing.division(mock_data, "income_log")
    
    # Assert
    assert processed_data.equals(expected_data)