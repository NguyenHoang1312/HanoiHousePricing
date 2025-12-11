import pandas as pd
import joblib as jl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# from feature_adder import FeatureAdder

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

rf = jl.load('results/linear_regression.joblib')
lr = jl.load('results/random_forest.joblib')

data = pd.read_csv('data/final_cleaned_dataset.csv')

X = data.iloc[:, 0: -1]
y = data.iloc[:, -1]

# Chia train/test như khi huấn luyện
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    shuffle=True,
    random_state=42
)

# --- Các chỉ số ---
lr = jl.load('results/linear_regression.joblib')
rf = jl.load('results/random_forest.joblib')

rf_y_pred = rf.predict(X_test)
lr_y_pred = lr.predict(X_test)

def calculate_score(actual, predict, model_name):
    mae = mean_absolute_error(actual, predict)
    mse = mean_squared_error(actual, predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predict)
    mape = np.mean(100 * np.abs(actual - predict)/actual)
    print(f'=== Model {model_name} ===')
    print(f'R2      : {r2:.6f}')
    print(f'MSE     : {mse:.6f}')
    print(f'RMSE    : {rmse:.6f}')
    print(f'MAE     : {mae:.6f}')
    print(f'MAPE    : {mape:.6f}')

calculate_score(y_test, lr_y_pred, 'Linear Regression')
calculate_score(y_test, rf_y_pred, 'Random Forest')

# --- Biểu đồ actual vs predict ---

y_test_reshaped = y_test.values.reshape(-1, 1)
lr_y_pred_reshaped = lr_y_pred.reshape(-1, 1)
rf_y_pred_reshaped = rf_y_pred.reshape(-1, 1)

lr_residual = y_test - lr_y_pred
rf_residual = y_test - rf_y_pred

lr_line = LinearRegression()
rf_line = LinearRegression()

lr_line.fit(y_test_reshaped, lr_y_pred_reshaped)
rf_line.fit(y_test_reshaped, rf_y_pred_reshaped)

x_line = np.linspace(y_test.min(), y_test.max(), 100).reshape(-1, 1)
lr_y_line = lr_line.predict(x_line)
rf_y_line = rf_line.predict(x_line)

lr_slope = lr_line.coef_[0][0]
lr_intercept = lr_line.intercept_[0]

rf_slope = rf_line.coef_[0][0]
rf_intercept = rf_line.intercept_[0]

# # Lấy step Random Forest
# rf_model = rf.named_steps['rf']
#
# # Lấy step preprocessor
# preprocessor = rf.named_steps['preprocessor']
#
# # Lấy tên các feature sau transform
# try:
#     feature_names_transformed = preprocessor.get_feature_names_out()
# except:
#     # Nếu TargetEncoder không hỗ trợ, fallback về cột numeric + encoded
#     feature_names_transformed = X_train.columns
# feature_names_clean = [name.split('__')[-1] for name in feature_names_transformed]
#
# # Lấy feature importance
# importances = rf_model.feature_importances_
#
# # Tạo DataFrame
# feat_imp_df = pd.DataFrame({
#     'feature': feature_names_clean,
#     'importance': importances
# })
#
# # Sắp xếp giảm dần và in top 10
# top_features = feat_imp_df.sort_values(by='importance', ascending=False).head(10)
# print(top_features)

_, axes = plt.subplots(1, 2, figsize=(10, 5))
font_size = 13
axes[0].set_title('Linear regression', fontsize=font_size)
axes[0].scatter(y_test, lr_y_pred, s=15, color='red')
axes[0].plot(x_line, lr_y_line, color='blue', label='đường hồi quy')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', linestyle='--', label='dự đoán hoàn hảo')
axes[0].set_xlabel('Giá nhà thực tế (triệu/m2)', fontsize=font_size)
axes[0].set_ylabel('Giá nhà dự đoán (triệu/m2)', fontsize=font_size)
axes[0].grid(True)
axes[0].legend(loc='upper left')

axes[1].set_title('Random Forest', fontsize=font_size)
axes[1].scatter(y_test, rf_y_pred, s=15, color='red')
axes[1].plot(x_line, rf_y_line, color='blue', label='đường hồi quy')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', linestyle='--', label='dự đoán hoàn hảo')
axes[1].set_xlabel('Giá nhà thực tế (triệu/m2)', fontsize=font_size)
axes[1].set_ylabel('Giá nhà dự đoán (triệu/m2)', fontsize=font_size)
axes[1].grid(True)
axes[1].legend(loc='upper left')

plt.tight_layout()
plt.show()

# --- slope và intercept ---
# print(f'Linear regression   : slope: {lr_slope:.4f} intercept: {lr_intercept:.4f}')
# print(f'Random forest       : slope: {rf_slope:.4f} intercept: {rf_intercept:.4f}')

# --- Biểu đồ predict vs redidual ---
# _, axes = plt.subplots(1, 2, figsize=(12, 5))
# fs = 13
# axes[0].scatter(lr_y_pred, lr_residual)
# axes[0].axhline(0, color='red', linestyle='--')
# axes[0].set_xlabel('Dự đoán (triệu/m2)', fontsize=fs)
# axes[0].set_ylabel('Residual (triệu/m2)', fontsize=fs)
# axes[0].set_title('Linear regression', fontsize=fs)
# axes[0].grid(True)
# axes[0].legend(loc='upper left')
#
# axes[1].scatter(rf_y_pred, rf_residual)
# axes[1].axhline(0, color='red', linestyle='--')
# axes[1].set_xlabel('Dự đoán (triệu/m2)', fontsize=fs)
# axes[1].set_ylabel('Residual (triệu/m2)', fontsize=fs)
# axes[1].set_title('Random forest', fontsize=fs)
# axes[1].grid(True)
# axes[1].legend(loc='upper left')
#
# plt.tight_layout()
# plt.show()

# --- Histogram sai số ---
# _, axes = plt.subplots(1, 2, figsize=(12, 5))
# fs = 13
# axes[0].hist(lr_residual, bins=32, color='turquoise', edgecolor='darkblue')
# axes[0].set_xlabel('Residual (triệu/m2)', fontsize=fs)
# axes[0].set_ylabel('Số lượng', fontsize=fs)
# axes[0].set_title('Linear regression', fontsize=fs)
# axes[0].grid(True)
# axes[0].legend(loc='upper left')
#
# axes[1].hist(rf_residual, bins=32, color='turquoise', edgecolor='darkblue')
# axes[1].set_xlabel('Residual (triệu/m2)', fontsize=fs)
# axes[1].set_ylabel('Số lượng', fontsize=fs)
# axes[1].set_title('Random Forest', fontsize=fs)
# axes[1].grid(True)
# axes[1].legend(loc='upper left')
#
# plt.tight_layout()
# plt.show()

# --- Dự đoán ---
# X_test_part = X_test.sample(n=5, random_state=13122004)
# y_test_part = y_test.sample(n=5, random_state=13122004)
# print(X_test_part)
# print(y_test_part)
#
# lr_y_pred_part = lr.predict(X_test_part)
# rf_y_pred_part = rf.predict(X_test_part)
#
# print(f'{lr_y_pred_part}')
# print(f'{rf_y_pred_part}')

