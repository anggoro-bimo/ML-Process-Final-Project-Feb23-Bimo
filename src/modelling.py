from sklearn.svm import SVC
from sklearn.metrics import classification_report
import sys
sys.path.append('/root/ml_process_feb23/')
import src.util as utils

def load_datasets(config: dict) -> pd.DataFrame:
    # Load every set of data
    x_train_ros = utils.pkl_load(config["dataset_train_balanced_path"][2])
    y_train_ros = utils.pkl_load(config["dataset_train_balanced_path"][3])

    x_valid = utils.pkl_load(config["dataset_valid_path"][0])
    y_valid = utils.pkl_load(config["dataset_valid_path"][1])

    x_test = utils.pkl_load(config["dataset_test_path"][0])
    y_test = utils.pkl_load(config["dataset_test_path"][1])

    return x_train_ros, y_train_ros, \
           x_valid, y_valid, \
           x_test, y_test,

def get_binned_features(dataset: pd.DataFrame):
    # Drop the features with original value
    
    dataset_bin = dataset.drop(config["original_cols"], axis = 1)
    return dataset_bin

def train_model(x_train_ros, y_train_ros, x_valid, y_valid):
    model = SVC(random_state = 23)
    model.fit(x_train_ros, y_train_ros)

    y_pred = model.predict(x_valid)
    print(classification_report(y_valid, y_pred))

    return model

if __name__ == "__main__" :
    # 1. Load config file
    config = utils.load_config()

    # 2. Load datasets
    x_train_ros, y_train_ros, \
    x_valid, y_valid, \
    x_test, y_test, = load_datasets(config)

    # 3. Include the binned features
    # 3.1. x_train
    x_train_ros_bin = get_binned_features(x_train_ros)
    # 3.2 x_valid
    x_valid_bin = get_binned_features(x_valid)
    # 3.2 x_test
    x_test_bin = get_binned_features(x_test)

    # 4. Train model
    model = train_model(x_train_ros, y_train_ros, x_valid, y_valid)

    # 5. Dump model
    utils.pkl_dump(best_model, config["production_model_path"])
