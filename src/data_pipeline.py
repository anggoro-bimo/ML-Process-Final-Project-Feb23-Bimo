import pandas as pd
import sys
PATH = "/home/er_bim/ML-Process-Final-Project-Feb23-Bimo/"
sys.path.append(PATH)
import src.util as utils
import copy
from sklearn.model_selection import train_test_split


def read_raw_data(config: dict) -> pd.DataFrame:
    # Return raw dataset
    credit_data = pd.read_csv(config["dataset_original_path"])
    return credit_data

def del_rows(dataset: pd.DataFrame, col: str, value: int) -> pd.DataFrame:
    dataset = dataset.loc[dataset[col] >= value].reset_index(drop = True)
    return dataset

def check_data(input_data: pd.DataFrame, config: dict, api: bool = False):
    input_data = copy.deepcopy(input_data)
    config = copy.deepcopy(config)

    if not api:
        # Check data types
        assert input_data.select_dtypes("object").columns.to_list() == \
            config["object_columns"], "input error, please fill the column(s) with 'yes' or 'no'."
        assert input_data.select_dtypes("int").columns.to_list() == \
            config["int_columns"], "input error, please fill the column(s) with any numeric character."
        assert input_data.select_dtypes("float").columns.to_list() == \
            config["float_columns"], "input error, please fill the column(s) with any numeric (decimal value is allowed) character."
    else:
        # In case checking data from api
        # Exclude the "card" feature, which is not a predictor
        object_columns = config["object_columns"]
        del object_columns[0] 

        # Check data types
        assert input_data.select_dtypes("object").columns.to_list() == \
            object_columns, "an error occurs in object column(s)."       

    assert set(input_data[config["object_columns"][1]]).issubset(set(config["range_owner"])), \
        "an error occurs in owner range."
    assert set(input_data[config["object_columns"][2]]).issubset(set(config["range_selfemp"])), \
        "an error occurs in selfemp range."
    assert input_data[config["int_columns"][0]].between(config["range_reports"][0], config["range_reports"][1]).sum() == \
        len(input_data), "an error occurs in reports range."
    assert input_data[config["float_columns"][0]].between(config["range_age"][0], config["range_age"][1]).sum() == \
        len(input_data), "an error occurs in age range."
    assert input_data[config["float_columns"][1]].between(config["range_income"][0], config["range_income"][1]).sum() == \
        len(input_data), "an error occurs in income range."
    assert input_data[config["float_columns"][2]].between(config["range_share"][0], config["range_share"][1]).sum() == \
        len(input_data), "an error occurs in share range."
    assert input_data[config["float_columns"][3]].between(config["range_expenditure"][0], config["range_expenditure"][1]).sum() == \
        len(input_data), "an error occurs in expenditure range."
    assert input_data[config["int_columns"][1]].between(config["range_dependents"][0], config["range_dependents"][1]).sum() == \
        len(input_data), "an error occurs in dependents range."    
    assert input_data[config["int_columns"][2]].between(config["range_months"][0], config["range_months"][1]).sum() == \
        len(input_data), "an error occurs in months range."
    assert input_data[config["int_columns"][3]].between(config["range_majorcards"][0], config["range_majorcards"][1]).sum() == \
        len(input_data), "an error occurs in majorcards range."
    assert input_data[config["int_columns"][4]].between(config["range_active"][0], config["range_active"][1]).sum() == \
        len(input_data), "an error occurs in active range."
   
def split_input_output(dataset: pd.DataFrame,
                       target_column: list,
                       save_file: bool = True,
                       return_file: bool = True):
    """Divide the data into its dependent variable/target (y-axis) and independent/predictor (x-axis) ones,
    input_df = predictors while output_df = target"""

    output_df = dataset[target_column]
    input_df = dataset.drop([target_column],
                            axis = 1)

    if save_file:  
        utils.pkl_dump(output_df, config["dataset_output_df_path"])
        utils.pkl_dump(input_df, config["dataset_input_df_path"])
    
    if return_file:
        return output_df, input_df

def split_train_test(x, y, TEST_SIZE):
    """Split the data into the training and test data,
    stratify parameter is activated,
    this function will be reproduced later as the data-splitting process further"""
    
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=TEST_SIZE,
                                                        random_state=46,
                                                        stratify=y)

    return x_train, x_test, y_train, y_test

def split_data(data_input: pd.DataFrame, 
               data_output: pd.DataFrame, 
               save_file: bool = True,
               return_file: bool = True, 
               TEST_SIZE: float = 0.17):
    """Split the data into the training, validation and test data,
    first split process will return into the train and test data,
    the train data resulted from the first split will be splitted further into the train and validation data.
    The TEST_SIZE 0.17 resulting the test data 17% proportion from the dataset length,
    while the proportion of train data and validation data are respectively 69% and 14%. 
    All files returned from this function are saved into pickle files."""

    x_train, x_test, y_train, y_test = split_train_test(data_input, 
                                                        data_output,
                                                        TEST_SIZE)

    x_train, x_valid, y_train, y_valid = split_train_test(x_train,
                                                          y_train,
                                                          TEST_SIZE)
    
    if save_file:
        utils.pkl_dump(x_train, config["dataset_train_path"][0])
        utils.pkl_dump(y_train, config["dataset_train_path"][1])
        utils.pkl_dump(x_valid, config["dataset_valid_path"][0])
        utils.pkl_dump(y_valid, config["dataset_valid_path"][1])
        utils.pkl_dump(x_test, config["dataset_test_path"][0])
        utils.pkl_dump(y_test, config["dataset_test_path"][1])

    if return_file:
        return x_train, y_train, \
            x_valid, y_valid, \
            x_test, y_test


if __name__ == "__main__":
    # 1. Loading configuration file
    config = utils.load_config()
    print("Configuration file loaded.")

    # 2. Loading dataset
    credit_data = read_raw_data(config)
    print("Raw dataset loaded.")

    # 3. Drop value <18 in feature 'age'
    credit_data = del_rows(credit_data, 'age', 18)
    print("Specific rows dropped.")

    # 4. Data Defense
    check_data(credit_data, config)
    print("Data defense mechanism activated.")

    # 5. Data Splitting, separate the predictors and target
    output_df, input_df = split_input_output(credit_data, target_column = "card", save_file = False)

    # 6. Data Splitting and saving as pickles
    x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(input_df, output_df)
    print("Raw dataset splitted and dumped as pickles. Ready to progress to data preprocessing.")