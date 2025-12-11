import pandas as pd
import streamlit as st
import os
import json
import joblib
from datetime import datetime
import requests

st.set_page_config(layout='wide')
st.title('DỰ ĐOÁN GIÁ NHÀ')
st.caption('Nhập thông tin về ngôi nhà và dự đoán giá bán tại thời điểm hiện tại.')
# API key
API_KEY_1 = 'xxx'
API_KEY_2 = 'xxx'
API_KEY = API_KEY_2

# Đọc path
hanoi_path = os.path.join(os.path.dirname(__file__), '..', 'datas', 'lookup', 'hanoi.json')
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest.joblib')
plan2_path = os.path.join(os.path.dirname(__file__), '..', 'datas', 'lookup', 'geocoding_plan2.csv')
history_path = os.path.join(os.path.dirname(__file__), '..', 'datas', 'lookup', 'predict_history.csv')

geocoding_plan2 = pd.read_csv(plan2_path)

# Lấy mapping huyện - xã
with open(hanoi_path, 'r', encoding='utf-8') as f:
    huyen_xa = json.load(f)

if 'huyen' not in st.session_state:
    st.session_state.huyen = list(huyen_xa.keys())[0]
if 'xa' not in st.session_state:
    st.session_state.xa = huyen_xa[st.session_state.huyen][0]

def update_huyen():
    st.session_state.xa = None

st.divider()

# Chia thành hai cột
col1, col2 = st.columns(2)
with col1:
    dien_tich = st.number_input('Diện tích (m²)', min_value=1.0, max_value=1000000.0, value=1.0)
    chieu_dai = st.number_input('Chiều dài (m)', min_value=1.0, max_value=1000.0, value=1.0)
    chieu_rong = st.number_input('Chiều rộng (m)', min_value=1.0, max_value=1000.0, value=1.0)
    so_tang = st.number_input('Số tầng', min_value=1, max_value=1000, value=3)
    so_phong_ngu = st.number_input('Số phòng ngủ', min_value=1, max_value=1000, value=1,)

with col2:
    loai_hinh_nha_o = st.selectbox('Loại hình nhà ở', options=[
        'Nhà ngõ, hẻm', 'Nhà mặt phố, mặt tiền', 'Nhà phố liền kề', 'Nhà biệt thự'
    ], index=0)
    giay_to_phap_ly = st.selectbox('Giấy tờ pháp lý', options=[
        'Đã có sổ', 'Đang chờ sổ', 'Giấy tờ khác'
    ], index=0)
    huyen = st.selectbox("Quận/Huyện", list(huyen_xa.keys()), index=list(huyen_xa.keys()).index(st.session_state.huyen))
    st.session_state.huyen = huyen
    xa = st.selectbox("Phường/Xã/Thị trấn", huyen_xa[st.session_state.huyen])
    st.session_state.xa = xa
    dia_chi = st.text_input('Địa chỉ (tuỳ chọn, số nhà, tên đường)', value='')

# Hàm geocoding
@st.cache_data
def geocoding(dia_chi, huyen, xa):
    is_successful_connection = False

    full_address = f'{dia_chi}, {xa}, {huyen}, Hà Nội'
    url = 'https://rsapi.goong.io/Geocode'
    params = {'address': full_address, 'api_key': API_KEY}

    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        if data.get('status') == 'OK' and data['results']:
            is_successful_connection = True
            loc = data['results'][0]['geometry']['location']
            return loc['lat'], loc['lng'], is_successful_connection
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        st.error('Timeout Error! Sử dụng dữ liệu dự phòng!')
        st.warning('Kiểm tra lại kết nối Internet và thử lại sau để có kết quả chính xác hơn!')
    except Exception as e:
        st.error(f'Lỗi: {e}')

    address = f'{xa}, {huyen}, Hà Nội'
    ss = geocoding_plan2.loc[geocoding_plan2['dia_chi'] == address]
    vi_do, kinh_do = None, None
    if not ss.empty:
        vi_do = ss['vi_do'].values[0]
        kinh_do = ss['kinh_do'].values[0]
    return vi_do, kinh_do, is_successful_connection

# Nhấn dự đoán
if st.button('**Dự đoán giá nhà**'):
    if chieu_dai < chieu_rong:
        st.error('Dữ liệu không hợp lệ!')
    else:
        is_successful_connection = False

        # Mã hóa địa chỉ sang tọa độ
        vi_do, kinh_do = 0.0, 0.0
        if dia_chi == '':
            address = f'{xa}, {huyen}, Hà Nội'
            ss = geocoding_plan2.loc[geocoding_plan2['dia_chi'] == address]
            if not ss.empty:
                vi_do = ss['vi_do'].values[0]
                kinh_do = ss['kinh_do'].values[0]
        else:
            vi_do, kinh_do, is_successful_connection = geocoding(dia_chi, huyen, xa)

        # Lấy thời gian hiện tại
        current_time = datetime.now()

        # Các đặc trưng
        feature = pd.DataFrame({
            'ngay': int(current_time.day),
            'thang': int(current_time.month),
            'nam': int(current_time.year),
            'xa': xa,
            'huyen': huyen,
            'vi_do': float(vi_do),
            'kinh_do': float(kinh_do),
            'loai_hinh_nha_o': loai_hinh_nha_o,
            'giay_to_phap_ly': giay_to_phap_ly,
            'so_tang': int(so_tang),
            'so_phong_ngu': int(so_phong_ngu),
            'dien_tich': float(dien_tich),
            'chieu_dai': float(chieu_dai),
            'chieu_rong': float(chieu_rong)
        }, index=[0])

        # Dự đoán và đưa ra kết quả
        model = joblib.load(model_path)
        gia_nha = model.predict(feature)
        st.success(f'**Giá nhà dự đoán: {gia_nha[0]:.2f} triệu/m².**')

        # Hiển thị bản đồ vị trí nhà
        coordinate_df = pd.DataFrame({
            'lat': [float(vi_do)],
            'lon': [float(kinh_do)]
        })
        st.map(coordinate_df)

        # Lưu lại kết quả dự đoán
        feature = feature.drop(columns=['ngay', 'thang', 'nam', 'vi_do', 'kinh_do'])
        feature.insert(loc=0, column='thoi_gian', value=f'{datetime.now().strftime("%H:%M:%S %d-%m-%Y")}')
        feature.insert(loc=1, column='dia_chi', value=f'{dia_chi if dia_chi != "" else "-"}')
        feature['ma_hoa_dia_chi'] = 'Goong Maps API' if is_successful_connection else 'Phương án dự phòng'
        feature['gia_nha'] = gia_nha
        feature.to_csv(
            history_path,
            mode='a',
            header=not os.path.exists(history_path),
            encoding='utf-8-sig',
            index=False
        )

st.info('Lưu ý: Kết quả mang tính tham khảo!')

