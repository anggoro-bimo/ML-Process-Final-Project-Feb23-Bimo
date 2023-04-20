import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import sys
sys.path.append('/root/ml_process_feb23/')
import src.util as utils

def load_datasets(config: dict) -> pd.DataFrame:
    # Load all the datasets needed

    df_train = utils.pkl_load(config['dataset_modelling_path'][0])
    x_train = df_train.drop(['card'], axis = 1)
    y_train = df_train['card']

    df_valid = utils.pkl_load(config['dataset_modelling_path'][1])
    x_valid = df_valid.drop(['card'], axis = 1)
    y_valid = df_valid['card']

    df_test = utils.pkl_load(config['dataset_modelling_path'][2])
    x_test = df_test.drop(['card'], axis = 1)
    y_test = df_test['card']

    return x_train, y_train, \
           x_valid, y_valid, \
           x_test, y_test,

def get_binned_features(dataset: pd.DataFrame, cols):
    # Drop the features with original value
    
    dataset_bin = dataset.drop(cols, axis = 1)
    return dataset_bin

def train_model(x_train_bin, y_train, \
                x_valid, y_valid, \
                x_test, y_test):
    
    best_model = DecisionTreeClassifier(random_state = 34)
    best_model.fit(x_train_bin, y_train)

    y_pred_on_valid = best_model.predict(x_valid_bin)
    print("The Classification Report and Confusion Matrix on validation dataset")
    print(classification_report(y_valid, y_pred_on_valid))

    y_pred_on_test = best_model.predict(x_test_bin)
    print("The Classification Report and Confusion Matrix on testing dataset")
    print(classification_report(y_test, y_pred_on_test))

    return best_model

if __name__ == "__main__" :
    # 1. Loading configuration file
    config = utils.load_config()
    print("Configuration file loaded.")

    # 2. Load datasets
    x_train, y_train, \
    x_valid, y_valid, \
    x_test, y_test, = load_datasets(config)
    print("Datasets loaded.")

    # 3. Include the binned features
    # 3.1. x_train
    x_train_bin = get_binned_features(x_train, config["original_cols"])
    # 3.2 x_valid
    x_valid_bin = get_binned_features(x_valid, config["original_cols"])
    # 3.2 x_test
    x_test_bin = get_binned_features(x_test, config["original_cols"])
    print("The binned features included to model training.")

    # 4. Train model
    best_model = train_model(x_train_bin, y_train, x_valid, y_valid, x_test, y_test)

    # 5. Dump model
    utils.pkl_dump(best_model, config["production_model_path"])
    print("The best model dumped, ready to progress to production.")
