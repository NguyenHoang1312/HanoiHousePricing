import streamlit as st
import pandas as pd
import os

st.set_page_config(layout='wide')
st.title('LỊCH SỬ DỰ ĐOÁN GIÁ NHÀ')

history_path = os.path.join(os.path.dirname(__file__), '..', 'datas', 'lookup', 'predict_history.csv')

def load_history() -> pd.DataFrame | None:
    if os.path.exists(history_path):
        try:
            return pd.read_csv(history_path)
        except Exception as e:
            st.error(f'Không đọc được dữ liệu lịch sử: {e}')
            return None
    return None

def clear_history():
    if os.path.exists(history_path):
        try:
            os.remove(history_path)
            st.success('Đã xoá lịch sử.')
        except Exception as e:
            st.error(f'Không thể xoá: {e}')
    else:
        st.info('Không có lịch sử để xoá.')
        
history = load_history()

if history is None or history.empty:
    st.info('Chưa có lịch sử dự đoán.')
else:
    st.subheader('Dữ liệu lịch sử')
    history_display = history.copy()
    history_display.columns = [
        'Thời gian dự đoán',
        'Địa chỉ nhà',
        'Xã',
        'Huyện',
        'Loại hình nhà ở',
        'Giấy tờ pháp lý',
        'Số tầng',
        'Số phòng ngủ',
        'Diện tích (m²)',
        'Chiều dài (m²)',
        'Chiều rộng (m²)',
        'Mã hóa địa chỉ',
        'Giá nhà dự đoán (m²)'
    ]
    st.dataframe(history_display)

    st.download_button(
        label='Tải xuống CSV',
        data=history.to_csv(index=False).encode('utf-8-sig'),
        file_name='predict_history.csv',
        mime='text/csv',
    )

st.divider()

if st.button('Xoá toàn bộ lịch sử', type='primary'):
    clear_history()