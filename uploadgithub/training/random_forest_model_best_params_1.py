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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATASET_PATH = 'data/cleaned_dataset_3.csv'
BEST_PARAMETERS_PATH = 'results/random_forest_best_parameters_1.joblib'

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
    ('cat', ce.TargetEncoder(cols=cat_cols, smoothing=1000), cat_cols),
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=42))
])

# Tìm best param lần 1 - RandomizedSearchCV
rf_parameters_grid_1 = {
    'rf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__max_depth': [int(x) for x in np.linspace(10, 50, num = 5)],
    'rf__min_samples_split': [5, 10, 15],
    'rf__min_samples_leaf': [5, 10, 15],
    'rf__max_samples': [0.6, 0.8, 1],
}

search_best_parameters_1 = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=rf_parameters_grid_1,
    scoring='neg_mean_absolute_error',
    n_iter=100,
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=2022
)

start_time = time.time()
search_best_parameters_1.fit(X_train, y_train)
searching_time = time.time() - start_time

print('searching time:', searching_time)
print('best params 1:', search_best_parameters_1.best_params_)
print('best r2 score 1:', search_best_parameters_1.best_score_)
print('r2 train 1:', search_best_parameters_1.score(X_train, y_train))
print('r2 test 1:', search_best_parameters_1.score(X_test, y_test))

# --- Lưu lại tham số tốt nhất ---
# jl.dump(search_best_parameters_1.best_params_, BEST_PARAMETERS_PATH)