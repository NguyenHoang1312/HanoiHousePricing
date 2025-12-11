import pandas as pd
import joblib as jl
import category_encoders as ce
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from feature_adder import  FeatureAdder

# --- Đọc dataset đã làm sạch ---
df = pd.read_csv('data/cleaned_dataset_3.csv')

X = df.iloc[:, 0: -1]
y = df.iloc[:, -1]

# Xóa đi 'dia_chi', 'huyen' và 'xa'
# Những thông tin này chỉ mang ý nghĩa phân tích
X = X.drop(columns=['dia_chi'], errors='ignore')

# --- Chia dữ liệu thành train/test (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    shuffle=True,
    random_state=42
)

# --- Pipeline ---
cat_cols = sorted(X.select_dtypes(include='object').columns.tolist())
num_cols = sorted(X.select_dtypes(include='number').columns.tolist())

preprocessor = ColumnTransformer([
    ('cat', ce.TargetEncoder(cols=cat_cols, smoothing=1000), cat_cols),
    ('num', StandardScaler(), num_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('add_features', FeatureAdder()),
    ('preprocessor', preprocessor),
    ('lr', LinearRegression(n_jobs=-1))
])
start_time = time.time()
pipeline.fit(X_train, y_train)
training_time = time.time() - start_time

# --- Lưu lại pipeline ---
jl.dump(pipeline, 'results/linear_regression.joblib')

print(f'Training time: {training_time:.6f}')