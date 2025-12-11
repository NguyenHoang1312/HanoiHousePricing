import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

df = pd.read_csv('data/cleaned_dataset.csv')

# Vị trí và giá nhà
# fig = px.scatter_mapbox(
#     df,
#     lat="vi_do", lon="kinh_do",
#     color="gia_nha",
#     size="gia_nha",
#     hover_name="huyen",
#     color_continuous_scale="Greens",
#     mapbox_style="carto-positron",
#     zoom=11,
#     center={"lat": 21.03, "lon": 105.83},
#     title="Bản đồ giá nhà Hà Nội (scatter overlay)"
# )
#
# fig.show()

# huyen_gia_nha = df.groupby('huyen')['gia_nha'].mean().reset_index()
# index = np.argsort(huyen_gia_nha['gia_nha'])[::-1]
# huyen = huyen_gia_nha['huyen'][index]
# gia_nha = huyen_gia_nha['gia_nha'][index]
#
# plt.barh(huyen, gia_nha,
#          color=plt.cm.Blues(gia_nha/gia_nha.max()),
#          edgecolor='black',
#          height=0.6)
# plt.xlabel('Giá nhà trung bình (triệu/m2)')
# plt.ylabel('Quận/huyện')

# Diện tích và giá nhà
# dien_tich = df['dien_tich'].values
# gia_nha = df['gia_nha'].values
#
# plt.scatter(dien_tich, gia_nha, s=30, color='blue')
# plt.xlabel('Diện tích (m2)')
# plt.ylabel('Giá nhà (triệu/m2)')

# Loại hình nhà ở và giá nhà
# a = df.groupby('loai_hinh_nha_o')['gia_nha'].mean().reset_index()
# loai_hinh_nha_o = a['loai_hinh_nha_o']
# gia_nha = a['gia_nha']
#
# plt.barh(loai_hinh_nha_o, gia_nha, color=plt.plasma(), height=0.6)
# plt.xlabel('Giá nhà trung bình (triệu/m2)')
# plt.ylabel('Loại hình nhà ở')

# Số tầng, số phòng ngủ và giá nhà
# so_tang = df['so_tang']
# so_phong_ngu = df['so_phong_ngu']
# gia_nha = df['gia_nha']
#
# _, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# axes[0].set_title('Số tầng và giá nhà')
# axes[0].scatter(so_tang, gia_nha, s=50, color='red')
# axes[0].set_xlabel('Số tầng')
# axes[0].set_ylabel('Giá nhà (triệu/m2)')
# axes[0].grid(True)
#
# axes[1].set_title('Số phòng ngủ và giá nhà')
# axes[1].scatter(so_phong_ngu, gia_nha, s=50, color='green')
# axes[1].set_xlabel('Số phòng ngủ')
# axes[1].set_ylabel('Giá nhà (triệu/m2)')
# axes[1].grid(True)

# Heat map
num_cols = df.select_dtypes(include='number').columns
df_corr = df[num_cols].corr()
plt.figure(figsize=(6 * 1.5, 5 * 1.5))
sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Ma trận tương quan giữa các đặc trưng số và giá nhà")

# plt.grid(True)
plt.tight_layout()
plt.show()


