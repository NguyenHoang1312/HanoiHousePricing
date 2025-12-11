import pandas as pd
import requests
import time
import os

# --- Cấu hình parameters ---
input_file  = 'data/cleaned_dataset_alt.csv'
output_file = 'data/geocoding_map.csv'
api_key_1   = 'xxx'    # Goong Maps API key
api_key_2   = 'xxx'
batch_size  = 1000                                          # Giới hạn 1000 requests/ngày 1 API key
batch_index = 1
api_key = api_key_1

# Ngày 20-10-2025:
#   + batch 0 (0-999)       (key 1): đã hoàn thành
#   + batch 1 (1000-1999)   (key 2): đã hoàn thành
# Ngày 21-10-2025:
#   + batch 2 (2000-2999)   (key 1): đã hoàn thành
#   + batch 3 (3000-3999)   (key 2): đã hoàn thành
# Ngày 22-10-2025:
#   + batch 4 (4000-4999)   (key 1): đã hoàn thành
#   + batch 5 (5000-5999)   (key 2): đã hoàn thành
# Ngày 23-10-2025:
#   + batch 6 (6000-6999)   (key 1): đã hoàn thành
#   + batch 7 (7000-7999)   (key 2): đã hoàn thành
# Ngày 24-10-2025:
#   + batch 8 (8000-8094)   (key 1): đã hoàn thành
#   + batch 1 (0-500)       (key 1): đã hoàn thành
#   + batch 2 ()

# --- Đọc dataset ---
data = pd.read_csv(input_file)

# --- Chia batch ---
start = batch_index * batch_size
end = min(start + batch_size, len(data))
subset = data.iloc[start:end].copy()

print(f'Batch {batch_index} ({start}–{end-1}), tổng {len(subset)} dòng')

# --- Tạo cột kết quả ---
subset['latitude'] = None
subset['longitude'] = None

# --- Hàm geocoding ---
def geocode(address):
    url = 'https://rsapi.goong.io/Geocode'
    params = {'address': address, 'api_key': api_key}
    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        if data.get('status') == 'OK' and data['results']:
            loc = data['results'][0]['geometry']['location']
            return loc['lat'], loc['lng']
    except Exception:
        print('Error:', Exception)
    return None, None

# --- Geocoding ---
for i, row in subset.iterrows():
    address = str(row['dia_chi'])
    lat, lng = geocode(address)
    subset.at[i, 'latitude'] = lat
    subset.at[i, 'longitude'] = lng
    print(f'{i+1} - {address} -> ({lat}, {lng})')
    time.sleep(0.2)

# --- Lưu kết quả ---
subset[['dia_chi', 'latitude', 'longitude']].to_csv(
    output_file,
    mode='a',
    header=not os.path.exists(output_file),
    index=False,
    encoding='utf-8-sig'
)

print(f'Lưu kết quả batch {batch_index} vào {output_file}')
