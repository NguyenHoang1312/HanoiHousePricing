import pandas as pd
import requests
import time
import os

# --- Cấu hình parameters ---
input_file  = 'data/cleaned_dataset_alt.csv'
output_file = 'data/geocoding_map.csv'
api_key_1   = 'gGOe1lpV33VWBjM06DKaVwcMfMwPcZgaZJuNd0sy'    # Goong Maps API key
api_key_2   = 'fIolR62rnpCRPFlT5PSWNkoBvegrBkTnEnRRtnNb'
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

"""
Module geocoding dùng Goong API.
- Hàm geocode(address, api_key_override=None) có thể import từ app Streamlit.
- Phần batch bên dưới chỉ chạy khi chạy trực tiếp file.
"""

# --- Hàm geocoding dùng cho cả import và script ---
def geocode(address, api_key_override=None):
    """Trả về (lat, lng) hoặc (None, None).
    - Ưu tiên api_key_override, nếu không có sẽ lấy từ biến môi trường GOONG_API_KEY,
      cuối cùng mới fallback về api_key ở đầu file.
    """
    key_to_use = api_key_override or os.getenv('GOONG_API_KEY') or api_key
    url = 'https://rsapi.goong.io/Geocode'
    params = {'address': address, 'api_key': key_to_use}
    try:
        res = requests.get(url, params=params, timeout=6)
        data = res.json()
        if data.get('status') == 'OK' and data.get('results'):
            loc = data['results'][0]['geometry']['location']
            return loc.get('lat'), loc.get('lng')
    except Exception as e:
        print('Error:', e)
    return None, None


if __name__ == '__main__':
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
