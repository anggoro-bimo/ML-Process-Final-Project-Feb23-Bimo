import numpy as np
import pandas as pd
import sys
sys.path.append('/root/ml_process_feb23/')
import src.util as utils
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def load_dataset(config: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = utils.pkl_load(config["dataset_train_path"][0])
    y_train = utils.pkl_load(config["dataset_train_path"][1])

    x_valid = utils.pkl_load(config["dataset_valid_path"][0])
    y_valid = utils.pkl_load(config["dataset_valid_path"][1])

    x_test = utils.pkl_load(config["dataset_test_path"][0])
    y_test = utils.pkl_load(config["dataset_test_path"][1])

    # Concatenate x and y each set
    df_train = pd.concat(
        [x_train, y_train],
        axis = 1
    )
    df_valid = pd.concat(
        [x_valid, y_valid],
        axis = 1
    )
    df_test = pd.concat(
        [x_test, y_test],
        axis = 1
    )

    # Return 3 set of data
    return df_train, df_valid, df_test

# Create the function for data transformation
def cols_transform(dataset, cols: list):
    """A function to transform the feature(s) value in the dataset into logarithmic value.
    The defined features are transformed and appended to the dataset,
    after the transformation and appendment of all features are done
    the features with original value are dropped from the dataset"""

    log_cols = []
    for i in range (len(cols)):
        col = cols[i]
        transformed = col + "_log" 
        dataset[transformed] = np.log10(dataset[col]+1)
        
        log_cols.append(transformed)
    dataset.drop(cols, axis = 1, inplace = True)
    return dataset

def label_encoding(dataset: pd.DataFrame, cols: list):
    """A function to convert the categorical values into the numeric ones.
    The list of categorical features are encoded by .map function,
    returning a dataset with the features encoded."""

    for i in range (len(cols)):
        col = cols[i]
        dataset[col] = dataset[col].map({'yes':1, 'no':0})
    return dataset

def binning(dataset: pd.DataFrame, col: str, bins: list, labels: list):
    """A function for categorize the feature's value into groups with specific interval defined in the bins parameter.
    Creates a new feature contains the value defined in the labels parameter."""
    
    binned = col +"_bin"
    dataset[binned] = pd.cut(dataset[col] , bins=bins, labels=labels, include_lowest=True).astype(int)
    return dataset

def division(dataset: pd.DataFrame, col: str):
    """A function to get the monthly value from any feature in the dataset that have annual value."""

    dataset[col] = dataset[col] / 12
    return dataset

def balancing(dataset: pd.DataFrame, 
              save_file: bool = True,
              return_file: bool = True):
    """A function to handle the dataset with imbalance target with various method from the imblearn library.
    x = the predictors in dataset, and y = the target in dataset,
    both method creates new x and y as data length changed,
    the new x and y dataframes are saved as pickle files"""

    y = dataset['card']
    x = dataset.drop(['card'], axis = 1)
    
    x_train_rus, y_train_rus = RandomUnderSampler(random_state = 46).fit_resample(x, y)
    x_train_ros, y_train_ros = RandomOverSampler(random_state = 85).fit_resample(x, y)

    if save_file:
        utils.pkl_dump(x_train_rus, config["dataset_train_balanced_path"][0])
        utils.pkl_dump(y_train_rus, config["dataset_train_balanced_path"][1])
        utils.pkl_dump(x_train_ros, config["dataset_train_balanced_path"][2])
        utils.pkl_dump(y_train_ros, config["dataset_train_balanced_path"][3])
        
    if return_file: 
        return x_train_rus, y_train_rus, \
               x_train_ros, y_train_ros


if __name__ == "__main__":

    # 1. Loading configuration file
    config = utils.load_config()
    print("Configuration file loaded.")

    # 2. Load datasets
    df_train, df_valid, df_test = load_dataset(config)
    print("Datasets loaded.")

    # 3. Data Transformation
    # 3.1. Training dataset
    df_train = cols_transform(df_train, config["cols_to_log"])
    # 3.2. Validation dataset
    df_valid = cols_transform(df_valid, config["cols_to_log"])
    # 3.1. Testing dataset
    df_test = cols_transform(df_test, config["cols_to_log"])
    print("Columns value transformed.")

    # 4. Label Encoding
    # 4.1. Training dataset
    df_train = label_encoding(df_train, config["cat_columns"])
    # 4.1. Validation dataset
    df_valid = label_encoding(df_valid, config["cat_columns"])
    # 4.1. Testing dataset
    df_test = label_encoding(df_test, config["cat_columns"])
    print("Columns value encoded.")

    # 5. Data Binning
    # 5.1. feature 'age'
    df_train = binning(df_train, 'age', config["bins_age"], config["labels_age"])
    df_valid = binning(df_valid, 'age', config["bins_age"], config["labels_age"])
    df_test = binning(df_test, 'age', config["bins_age"], config["labels_age"])
    # 5.2. feature 'reports'
    df_train = binning(df_train, 'reports', config["bins_reports"], config["labels_reports"])
    df_valid = binning(df_valid, 'reports', config["bins_reports"], config["labels_reports"])
    df_test = binning(df_test, 'reports', config["bins_reports"], config["labels_reports"])
    # 5.3. feature 'dependents'
    df_train = binning(df_train, 'dependents', config["bins_dependents"], config["labels_dependents"])
    df_valid = binning(df_valid, 'dependents', config["bins_dependents"], config["labels_dependents"])
    df_test = binning(df_test, 'dependents', config["bins_dependents"], config["labels_dependents"])
    # 5.1. feature 'active'
    df_train = binning(df_train, 'active', config["bins_active"], config["labels_active"])
    df_valid = binning(df_valid, 'active', config["bins_active"], config["labels_active"])
    df_test = binning(df_test, 'active', config["bins_active"], config["labels_active"])
    print("Columns value binned.")

    # 6. Value Division on the 'income_log' feature
    # 6.1. Training dataset
    df_train = division(df_train, 'income_log')
    # 6.1. Training dataset
    df_valid = division(df_valid, 'income_log')
    # 6.1. Training dataset
    df_test = division(df_test, 'income_log')
    print("Column value divided.")

    # 7. Dump the datasets
    utils.pkl_dump(df_train, config["dataset_modelling_path"][0])
    utils.pkl_dump(df_valid, config["dataset_modelling_path"][1])
    utils.pkl_dump(df_test, config["dataset_modelling_path"][2])
    print("Datasets dumped.")
    
    # 8. Data Balancing
    x_train_rus, y_train_rus, x_train_ros, y_train_ros = balancing(df_train)
    print("Imbalance data treated, datasets dumped as pickles. Ready to progress to modelling.")
    