import streamlit as st

home = st.Page('pages/home.py', title='Trang chủ', icon='🏠')
dataset = st.Page('pages/dataset.py', title='Dữ liệu thống kê', icon='📈')
predict =  st.Page('pages/predict.py', title='Dự đoán giá nhà', icon='📝')
history = st.Page('pages/history.py', title='Lịch sử dự đoán', icon='⌛')

nav = st.navigation([home, dataset, predict, history])
nav.run()
