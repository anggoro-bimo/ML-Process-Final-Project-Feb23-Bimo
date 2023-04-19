import pandas as pd
import sys
sys.path.append('/root/ml_process_feb23/')
import src.util as utils
import preprocessing

def test_label_encoding():
    # Arrange
    config = utils.load_config()

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
    expected_data = expected_data.astype(int)
     
    # Act
    processed_data = preprocessing.label_encoding(mock_data)

    # Assert
    assert processed_data.equals(expected_data)
