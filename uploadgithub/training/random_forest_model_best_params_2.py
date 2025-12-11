import pandas as pd
import time
import numpy as np
import joblib as jl
import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATASET_PATH = 'data/cleaned_dataset_3.csv'
BEST_PARAMETERS_1_PATH = 'results/random_forest_best_parameters_1.joblib'
BEST_PARAMETERS_2_PATH = 'results/random_forest_best_parameters_2_2.joblib'

# --- Đọc dữ liệu ---
df = pd.read_csv(DATASET_PATH)

X = df.iloc[:, 0 :-1]
y = df.iloc[:, -1]

# Xóa đi 'dia_chi', 'huyen' và 'xa'
X = X.drop(columns=['dia_chi'], errors='ignore')

# --- Chia dữ liệu thành train/test (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

# --- pipeline ---
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

preprocessor = ColumnTransformer([
    # ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ('cat', ce.TargetEncoder(cols=cat_cols, smoothing=1000), cat_cols),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=42))
])

search_best_parameters_1 = jl.load(BEST_PARAMETERS_1_PATH)
rf_best_params = search_best_parameters_1

# Tìm best param lần 2 - GridSearchCV
rf_parameters_grid_2 = {
    'rf__n_estimators':[
        max(10, rf_best_params['rf__n_estimators'] - 100),
        rf_best_params['rf__n_estimators'],
        rf_best_params['rf__n_estimators'] + 100
    ],
    'rf__max_features': ['sqrt', 'log2', 1/3],
    'rf__max_depth': [
        max(1, rf_best_params['rf__max_depth'] - 10),
        rf_best_params['rf__max_depth'],
        rf_best_params['rf__max_depth'] + 10
    ],
    'rf__min_samples_split': [5, 10, 15],
    'rf__min_samples_leaf': [5, 10, 15],
    'rf__max_samples': [0.6, 0.8, 1.0]
}

search_best_parameters_2 = GridSearchCV(
    estimator=pipeline,
    param_grid=rf_parameters_grid_2,
    scoring='neg_mean_absolute_error',
    cv=3,
    n_jobs=-1,
    verbose=2
)

start_time = time.time()
search_best_parameters_2.fit(X_train, y_train)
searching_time = time.time() - start_time

print('searching time:', searching_time)

print('best params 2:', search_best_parameters_2.best_params_)
print('best r2 score 2:', search_best_parameters_2.best_score_)
print('r2 train 2:', search_best_parameters_2.score(X_train, y_train))
print('r2 test 2:', search_best_parameters_2.score(X_test, y_test))

# --- Lưu lại tham số tốt nhất ---
# jl.dump(search_best_parameters_2.best_params_, BEST_PARAMETERS_2_PATH)