# file này dành cho giai đoạn tiền xử lý dữ liệu

import numpy as np
import pandas as pd
import unidecode
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)

# --- Đọc dữ liệu từ dataset.csv ---
house_prices = pd.read_csv('data/dataset.csv')

# --- Xóa các cột không sử dụng ---
house_prices = house_prices.drop(columns='Unnamed: 0', axis=1)
house_prices = house_prices.drop(columns='Ngày', axis=1)

# Đổi tên toàn bộ các cột về tiếng Việt không dấu, ngăn cách bằng dấu '_'
house_prices.columns = [
    unidecode.unidecode(col).lower().replace(' ', '_')
    for col in house_prices.columns
]

house_prices = house_prices.rename(columns={'gia/m2': 'gia_nha'})
house_prices = house_prices.rename(columns={'dai': 'chieu_dai'})
house_prices = house_prices.rename(columns={'rong': 'chieu_rong'})

# --- Loại bỏ bản ghi trùng lặp ---
house_prices = house_prices.drop_duplicates()

# --- Loại bỏ các bản ghi chứa giá trị NaN ---
house_prices = house_prices.dropna()

# --- Xử lý số tầng và số phòng ngủ ---
house_prices = house_prices[~(house_prices['so_tang'] == 'Nhiều hơn 10')]
house_prices = house_prices[~(house_prices['so_phong_ngu'] == 'nhiều hơn 10 phòng')]

house_prices['so_tang'] = house_prices['so_tang'].astype(int)
house_prices['so_phong_ngu'] = house_prices['so_phong_ngu'].str.replace(' phòng', '')
house_prices['so_phong_ngu'] = house_prices['so_phong_ngu'].astype(int)

# --- Loại bỏ đơn vị tại các cột diện tích, dài và rộng ---
house_prices['chieu_dai'] = (house_prices['chieu_dai']
                             .str.replace('m', '')
                             .str.strip().astype(float))
house_prices['chieu_rong'] = (house_prices['chieu_rong']
                              .str.replace('m', '')
                              .str.strip().astype(float))
house_prices['dien_tich'] = (house_prices['dien_tich']
                             .str.replace('m²', '')
                             .str.strip().astype(float))

# --- Xử lý các giá trị ngoại lai, thường do lỗi nhập liệu ---
def remove_outlier(data_frame, series):
    q1 = data_frame[series].quantile(0.25)
    q3 = data_frame[series].quantile(0.75)
    iqr = q3 - q1
    condition = (data_frame[series] < (q1 - 1.5 * iqr)) | (data_frame[series] > (q3 + 1.5 * iqr))
    removed_data_frame = data_frame[~condition]
    return removed_data_frame

columns_with_outliers = ['dien_tich', 'chieu_dai', 'chieu_rong']
for col in columns_with_outliers:
    house_prices = remove_outlier(house_prices, col)

# --- Xử lý giá nhà ---
# Loại bỏ các căn nhà có giá đ/m², đây là lỗi nhập liệu
house_prices = house_prices[~(house_prices['gia_nha'].str.contains('đ/m²'))]

# Chỉ có hai loại đơn vị triệu và tỷ, đưa toàn bộ về triệu/m² và chuyển về float
mark = house_prices['gia_nha'].str.contains('tỷ/m²')
house_prices['gia_nha'] = (house_prices['gia_nha'].str.replace(r'(triệu|tỷ)/m²', '', regex=True)
                                                  .str.strip())
house_prices['gia_nha'] = house_prices['gia_nha'].str.replace('.', '')
house_prices['gia_nha'] = (house_prices['gia_nha'].str.replace(',', '.')
                                                  .str.strip().astype(float))
house_prices.loc[mark, 'gia_nha'] = house_prices.loc[mark, 'gia_nha'] * 1000

# --- Xử lý địa chỉ ---
# Loại bỏ những địa chỉ quá ngắn hoặc địa chỉ không có dấu ',' nhưng quá dài
mask = (((house_prices['dia_chi'].str.split(',').str.len() == 1) &
         (house_prices['dia_chi'].str.split(' ').str.len() > 5)) |
        (house_prices['dia_chi'].str.len() < 5))
house_prices = house_prices[~mask]

# Những địa chỉ ngắn còn lại chỉ có (số) tên đường, tiến hành ghép tên đầy đủ
def merge_address(data_frame, address_col, commune_col, district_col):
    address = data_frame[address_col]
    commune = data_frame[commune_col]
    district = data_frame[district_col]
    merged_address = address + ', ' + commune + ', ' + district + ', Hà Nội'

    conditions = (data_frame[address_col].str.split(',').str.len() == 1)
    data_frame.loc[conditions, address_col] = merged_address
    return data_frame

house_prices = merge_address(house_prices, 'dia_chi', 'xa', 'huyen')

# --- Lưu lại dataset đã được làm sạch ---
house_prices.to_csv(
    'data/final_cleaned_dataset.csv',
    sep=',',
    encoding='utf-8-sig',
    index=False,
    header=True
)