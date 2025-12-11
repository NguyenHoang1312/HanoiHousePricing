import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Đọc dữ liệu đã làm sạch ---
house_prices = pd.read_csv('data/final_cleaned_dataset.csv')

# --- Giá nhà ---
# gia_nha = house_prices['gia_nha'].values
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# axes[0].hist(gia_nha, bins=32, color='pink', edgecolor='red')
# axes[0].set_title('Histogram giá nhà')
# axes[0].set_xlabel('Giá nhà (triệu/m2)')
# axes[0].set_ylabel('Số lượng')
# axes[0].grid(True)

# axes[1].boxplot(gia_nha)
# axes[1].set_title('Boxplot giá nhà')
# axes[1].set_ylabel('Giá nhà (triệu/m2)')
# axes[1].grid(True)

# axes[0].boxplot(gia_nha)
# axes[0].set_title('Histogram của giá nhà ban đầu')
# axes[0].set_ylabel('Giá nhà (triệu/m2)')
# axes[1].grid(True)
#
# axes[1].boxplot(outlier_gia_nha)
# axes[1].set_title('Histogram của giá nhà đã xử lý ngoại lai')
# axes[1].set_ylabel('Giá nhà (triệu/m2)')
# axes[1].grid(True)

# --- Huyện ---
# huyen = house_prices['huyen']
# counts = huyen.value_counts()
# labels = counts.index
#
# plt.barh(labels, counts, color='skyblue', edgecolor='black')
# plt.xlabel('Số căn nhà được khảo sát')
# plt.ylabel('Huyện')
# plt.title('Số lượng căn nhà được khảo sát theo quận, huyện')
# plt.grid(True)

# --- Giấy tờ pháp lý ---
# giay_to = house_prices['giay_to_phap_ly']
# counts = giay_to.value_counts()
# labels = counts.index
# percentages = counts / counts.sum() * 100
# legend_labels = [f'{label} ({perc:.1f}%)' for label, perc in zip(labels, percentages)]
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# axes[0].bar(labels, counts, color='pink', edgecolor='red')
# axes[0].set_title('Số lượng căn nhà theo tình trạng giấy tờ')
# axes[0].set_xlabel('Tình trạng giấy tờ pháp lý')
# axes[0].set_ylabel('Số lượng')
# axes[0].grid(True)
#
# axes[1].pie(counts, startangle=90)
# axes[1].set_title('Tỉ lệ căn nhà theo tình trạng giấy tờ')
# axes[1].grid(True)
# axes[1].legend(legend_labels, title='Tình trạng giấy tờ', loc="center left", bbox_to_anchor=(1, 0.5))

# --- Loại hình nhà ở ---
# loai_hinh_nha_o = house_prices['loai_hinh_nha_o']
# counts = loai_hinh_nha_o.value_counts()
# labels = counts.index
# percentages = counts / counts.sum() * 100
# legend_labels = [f'{label} ({perc:.1f}%)' for label, perc in zip(labels, percentages)]
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# axes[0].bar(labels, counts, color='blue', edgecolor='violet')
# axes[0].set_title('Số lượng theo loại hình nhà ở')
# axes[0].set_xlabel('Loại hình nhà ở')
# axes[0].set_ylabel('Số lượng')
# axes[0].grid(True)
#
# axes[1].pie(counts, startangle=90)
# axes[1].set_title('Tỉ lệ theo loại hình nhà ở')
# axes[1].grid(True)
# axes[1].legend(legend_labels, title='Loại hình nhà ở', loc="center left", bbox_to_anchor=(1, 0.5))

# --- Số tầng ---
# so_tang = house_prices['so_tang']
#
# _, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# axes[0].hist(so_tang, bins=32, color='lime', edgecolor='black')
# axes[0].set_title('Histogram của số tầng')
# axes[0].set_xlabel('Số tầng')
# axes[0].set_ylabel('Số lượng')
# axes[0].grid(True)
#
# axes[1].boxplot(so_tang)
# axes[1].set_title('Boxplot của số tầng')
# axes[1].set_ylabel('Số tầng')
# axes[1].grid(True)

# --- Số phòng ngủ ---
# so_tang = house_prices['so_phong_ngu']
# print(so_tang.unique())
# _, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# axes[0].hist(so_tang, bins=32, color='pink', edgecolor='purple')
# axes[0].set_title('Histogram của số phòng ngủ')
# axes[0].set_xlabel('Số phòng ngủ')
# axes[0].set_ylabel('Số lượng')
# axes[0].grid(True)
#
# axes[1].boxplot(so_tang)
# axes[1].set_title('Boxplot của số phòng ngủ')
# axes[1].set_ylabel('Số phòng ngủ')
# axes[1].grid(True)

# --- Diện tích ---
so_tang = house_prices['dien_tich']
print(so_tang.unique())
_, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(so_tang, bins=32, color='gold', edgecolor='black')
axes[0].set_title('Histogram của diện tích')
axes[0].set_xlabel('Diện tích (m2)')
axes[0].set_ylabel('Số lượng')
axes[0].grid(True)

axes[1].boxplot(so_tang)
axes[1].set_title('Boxplot của diện tích')
axes[1].set_ylabel('Số phòng ngủ')
axes[1].grid(True)

plt.tight_layout()
plt.show()