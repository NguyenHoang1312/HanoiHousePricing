import pandas as pd
import joblib as jl
import numpy as np
import category_encoders as ce
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from feature_adder import  FeatureAdder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET_PATH = 'data/cleaned_dataset_3.csv'
BEST_PARAMETERS_PATH = 'results/random_forest_best_parameters_2_2.joblib'
MODEL_PATH = 'results/random_forest.joblib'

# --- Gọi best params ---
best_parameters = jl.load(BEST_PARAMETERS_PATH)
rf_parameters = {key.replace('rf__', ''): value for key, value in best_parameters.items()}

# --- Đọc dataset đã làm sạch ---
df = pd.read_csv('data/cleaned_dataset_3.csv')

X = df.iloc[:, 0: -1]
y = df.iloc[:, -1]

# Xóa đi 'dia_chi', 'huyen' và 'xa'
# Những thông tin này chỉ mang ý nghĩa phân tích
print(X.shape)
X = X.drop(columns=['dia_chi'], errors='ignore')
print(X.shape)
# --- Chia dữ liệu thành train/test (80/20) ---
rs = int(np.random.randint(100, 999))
# rs = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    shuffle=True,
    random_state=rs
)


# --- Pipeline ---
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

# === LINEAR REGRESSION ===
lr_preprocessor = ColumnTransformer([
    ('cat', ce.TargetEncoder(cols=cat_cols, smoothing=1000), cat_cols),
    ('num', StandardScaler(), num_cols)
], remainder='passthrough')

lr_pipeline = Pipeline([
    ('add_features', FeatureAdder()),
    ('preprocessor', lr_preprocessor),
    ('lr', LinearRegression(n_jobs=-1))
])
lr_start_time = time.time()
lr_pipeline.fit(X_train, y_train)
lr_training_time = time.time() - lr_start_time

# --- Lưu lại pipeline ---
# jl.dump(pipeline, 'results/linear_regression.joblib')

# === RANDOM FOREST ===

rf_preprocessor = ColumnTransformer([
    ('cat', ce.TargetEncoder(cols=cat_cols, smoothing=1000), cat_cols),
], remainder='passthrough')

rf_pipeline = Pipeline([
    ('add_features', FeatureAdder()),
    ('preprocessor', rf_preprocessor),
    ('rf', RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True, **rf_parameters))
])

# Huấn luyện mô hình
rf_start_time = time.time()
rf_pipeline.fit(X_train, y_train)
rf_training_time = time.time() - rf_start_time

# Lưu lại mô hình
# jl.dump(rf_pipeline, MODEL_PATH)

# === RESULTS ===

lr_y_pred = lr_pipeline.predict(X_test)
rf_y_pred = rf_pipeline.predict(X_test)

def calculate_score(actual, predict, model_name):
    mae = mean_absolute_error(actual, predict)
    mse = mean_squared_error(actual, predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predict)
    mape = np.mean(100 * np.abs(actual - predict)/actual)
    print(f'=== Model {model_name} ===')
    print(f'R2      : {r2:.2f}')
    print(f'MSE     : {mse:.2f}')
    print(f'RMSE    : {rmse:.2f}')
    print(f'MAE     : {mae:.2f}')
    print(f'MAPE    : {mape:.2f}')

print(f'Random state: {rs}')
calculate_score(y_test, lr_y_pred, 'Linear Regression')
print(f'Training time LR: {lr_training_time:.2f}')
calculate_score(y_test, rf_y_pred, 'Random Forest')
print(f'Traing time RF: {rf_training_time:.2f}s')
