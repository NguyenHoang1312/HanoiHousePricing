import streamlit as st
import os
import base64

st.set_page_config(layout='wide')

img_path = os.path.join(os.path.dirname(__file__), "..", "assets", "background1.png")
with open(img_path, "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

st.markdown(f"""
    <div style="
        background: url(data:image/png;base64,{encoded});
        background-size: cover;
        background-position: center;
        padding: 2rem;
        border-radius: 15px;">
        <h2 style="color: white; font-size: 60px;">PHÂN TÍCH DỮ LIỆU VÀ DỰ ĐOÁN GIÁ NHÀ HÀ NỘI</h2>
        <p style="color: white;">Đồ án chuyên ngành khoa học máy tính.</p>
        <p style="color: white; font-weight: bold;">Nhóm 02 - Lớp 20251IT6052002</p>
    </div>
    """, unsafe_allow_html=True)
st.write('')
st.info('Ứng dụng hỗ trợ phân tích dữ liệu giá nhà, hỗ trợ các biểu đồ trực quan. '
            '\nHỗ trợ dự đoán giá nhà tại Hà Nội nhanh chóng, chính xác cùng mô hình Random Forest.')

with st.form(key='infor'):
    st.header('Dữ liệu:')
    st.write('☞ Tải dữ liệu: tải tệp csv từ thiết bị của bạn.')
    st.write('☞ Xem nhanh các mô tả: hiển thị ngay lập thức khi tải dữ liệu.')
    st.write('☞ Các biểu đồ chi tiết: đầy đủ từ đơn biến đến đa biến, hiểu rõ hơn về dữ liệu.')
    st.header('Dự đoán giá nhà:')
    st.write('☞ Dự đoán nhanh chóng: nhập thông số và một nút nhấn là có ngay giá nhà dự đoán tại thời điểm hiện tại.')
    st.write('☞ Lưu lịch sử: cho phép xem lại dữ liệu đã dự đoán, tham khảo và so sánh.')

    st.divider()

    summited = st.form_submit_button('**Dự đoán giá nhà ngay!**', type='primary')
    if summited:
        st.switch_page('pages/predict.py')

