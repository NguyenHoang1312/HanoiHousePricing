import pandas as pd
import joblib as jl
import category_encoders as ce
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from feature_adder import FeatureAdder

DATASET_PATH = 'data/cleaned_dataset_3.csv'
BEST_PARAMETERS_PATH = 'results/random_forest_best_parameters_2_2.joblib'
MODEL_PATH = 'results/random_forest.joblib'

# --- Đọc dữ liệu ---
df = pd.read_csv(DATASET_PATH)

X = df.iloc[:, 0 :-1]
y = df.iloc[:, -1]

# Xóa đi 'dia_chi'
X = X.drop(columns=['dia_chi'], errors='ignore')

# --- Chia dữ liệu thành train/test (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    shuffle=True,
    random_state=42
)

# --- Gọi best params ---
best_parameters = jl.load(BEST_PARAMETERS_PATH)
rf_parameters = {key.replace('rf__', ''): value for key, value in best_parameters.items()}
print(rf_parameters)
# --- pipeline ---
cat_cols = sorted(X.select_dtypes(include='object').columns.tolist())
num_cols = sorted(X.select_dtypes(include='number').columns.tolist())

preprocessor = ColumnTransformer([
    ('cat', ce.TargetEncoder(cols=cat_cols, smoothing=1000), cat_cols),
], remainder='passthrough')

pipeline=Pipeline([
    ('add_features', FeatureAdder()),
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True, **rf_parameters))
])

# Huấn luyện mô
start_time = time.time()
pipeline.fit(X_train, y_train)
training_time = time.time() - start_time

# Lưu lại mô hình
jl.dump(pipeline, MODEL_PATH)

print(f'Traing time: {training_time:.6f}s')