from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Data paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "example_extracted_text.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models" / "saved_models"
LOGS_DIR = ROOT_DIR / "logs"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

def create_ann_model():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(1678,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Model configurations
MODEL_CONFIGS = {
    'random_forest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE),
        'param_grid': {
            'n_estimators': [200, 300],
            'max_depth': [10, 15],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [4, 8],
            'class_weight': ['balanced'],
            'max_features': ['sqrt', 'log2']
        }
    },
    'svm': {
        'model': SVC(random_state=RANDOM_STATE, probability=True),
        'param_grid': {
            'C': [0.01, 0.1, 1.0],
            'kernel': ['rbf'],
            'gamma': ['scale'],
            'class_weight': ['balanced'],
            'probability': [True]
        }
    },
    'mlp': {
        'model': MLPClassifier(random_state=RANDOM_STATE),
        'param_grid': {
            'hidden_layer_sizes': [(100, 50), (100, 50, 25), (200, 100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.1, 0.5, 1.0],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.001, 0.01],
            'early_stopping': [True],
            'validation_fraction': [0.2],
            'n_iter_no_change': [30],
            'max_iter': [3000],
            'batch_size': [64, 128],
            'solver': ['adam']
        }
    },
    'logistic': {
        'model': LogisticRegression(random_state=RANDOM_STATE),
        'param_grid': {
            'C': [0.001, 0.01, 0.1, 1.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'class_weight': ['balanced'],
            'max_iter': [2000],
            'tol': [1e-4]
        }
    },
    'catboost': {
        'model': CatBoostClassifier(random_state=RANDOM_STATE),
        'param_grid': {
            'learning_rate': [0.01, 0.1],
            'depth': [6, 8],
            'l2_leaf_reg': [1, 3],
            'iterations': [200, 300],
            'early_stopping_rounds': [20],
            'eval_metric': ['AUC'],
            'loss_function': ['Logloss']
        }
    }
}

# Updated ensemble weights to include CatBoost
ENSEMBLE_WEIGHTS = {
    'random_forest': 0.25,
    'logistic': 0.20,
    'svm': 0.20,
    'mlp': 0.15,
    'catboost': 0.20
}

# MLflow settings
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "text_classification" 