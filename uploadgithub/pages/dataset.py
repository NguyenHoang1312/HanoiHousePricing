# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import json
# from utils.preprocess import preprocess
#
# # file cần dùng
# mapping_path = os.path.join(os.path.dirname(__file__), '..', 'datas', 'lookup', 'column_names.json')
# with open(mapping_path, 'r', encoding='utf-8-sig') as f:
#     column_name = json.load(f)
#
# # Tiêu đề
# st.set_page_config(layout='wide')
# st.title('DỮ LIỆU THỐNG KÊ')
# st.caption('Tải lên một tệp CSV để xem nhanh dữ liệu, thống kê mô tả và biểu đồ.')
#
# uploaded = st.file_uploader('Chọn tệp CSV', type=['csv'])
# if uploaded is not None and 'df' not in st.session_state:
#     st.session_state['df'] = pd.read_csv(uploaded)
#
# if 'df' in st.session_state:
#     try:
#         df = st.session_state['df']
#     except Exception as e:
#         st.error(f'Không thể đọc tệp CSV: {e}')
#         df = None
#
#     if df is not None and df.shape[0] > 0:
#         # Tiền xử lý
#         df = preprocess(df)
#
#         # Xem nhanh dữ liệu
#         st.subheader('Xem nhanh dữ liệu')
#         df_display = df.copy()
#         df_display = df_display.rename(columns=column_name)
#         df_display = df_display.reset_index(drop=True)
#         st.dataframe(df_display.head(20))
#
#         # Thống kê mô tả cơ bản cho cột số
#         numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#         numeric_cols_display = [column_name[col] for col in numeric_cols]
#         if len(numeric_cols) > 0:
#             st.subheader('Thống kê mô tả')
#             desc = df[numeric_cols].describe().T
#             # Bổ sung median (50%) theo tên rõ ràng
#             desc['median'] = df[numeric_cols].median()
#             desc.index = numeric_cols_display
#             st.dataframe(desc)
#         else:
#             st.info('Không phát hiện cột số để tính thống kê mô tả.')
#
#         # Biểu đồ tự động
#         st.subheader('Biểu đồ đơn biến')
#
#         with st.expander('Biểu đồ cho cột số', expanded=False):
#             def show_single_numeric_cols(col):
#                 fig, ax = plt.subplots(1, 2, figsize=(8, 3))
#                 ax[0].hist(df[col], bins=32, color='skyblue', edgecolor='blue')
#                 ax[0].set_title(f'Histogram của {column_name[col]}')
#                 ax[0].set_xlabel(column_name[col])
#                 ax[0].set_ylabel('Tần suất')
#                 ax[0].grid(True, alpha=0.3)
#                 ax[1].boxplot(df[col])
#                 ax[1].set_title(f'Boxplot của {column_name[col]}')
#                 ax[1].set_ylabel(column_name[col])
#                 ax[1].grid(True)
#
#                 st.pyplot(fig)
#             if len(numeric_cols) == 0:
#                 st.write('Không có cột số để vẽ histogram.')
#             else:
#                 for col in numeric_cols:
#                     show_single_numeric_cols(col)
#
#         with st.expander('Biểu đồ cho cột phân loại', expanded=False):
#             def show_single_categorical_cols(col):
#                 series_values = df[col]
#                 counts = series_values.value_counts()
#                 labels = counts.index
#                 percentages = counts / counts.sum() * 100
#                 legend_labels = [f'{label} ({perc:.1f}%)' for label, perc in zip(labels, percentages)]
#                 fig, ax = plt.subplots(1, 2, figsize=(8, 3))
#                 ax[0].barh(counts.index.astype(str), counts.values, color='gold', edgecolor='red')
#                 ax[0].set_title(f'Số lượng căn nhà theo {column_name[col]}')
#                 ax[0].set_xlabel(f'{column_name[col]}')
#                 ax[0].set_ylabel('Số lượng')
#                 ax[0].grid(True, axis='x', alpha=0.3)
#                 ax[1].pie(counts, startangle=90)
#                 ax[1].set_title(f'Tỉ lệ căn nhà theo {column_name[col]}')
#                 ax[1].grid(True)
#                 ax[1].legend(legend_labels, title=column_name[col], loc="center left", bbox_to_anchor=(1, 0.5))
#                 st.pyplot(fig)
#
#             # categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#             categorical_cols = ['huyen', 'loai_hinh_nha_o']
#             if len(categorical_cols) == 0:
#                 st.write('Không có cột phân loại để vẽ bar chart.')
#             else:
#                 for col in categorical_cols:
#                     show_single_categorical_cols(col)
#
#         st.subheader('Biểu đồ đa biến')
#         with st.expander('Tương quan biến số và giá nhà', expanded=False):
#             def show_heatmap():
#                 dfcorr = df_display[numeric_cols_display].corr()
#
#                 fig, ax = plt.subplots(figsize=(8, len(numeric_cols) * 0.4))
#                 sns.heatmap(dfcorr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
#                 ax.set_title('Mức độ tương quan các đặc trưng số')
#                 st.pyplot(fig)
#
#             if 'gia_nha' in df.columns:
#                 show_heatmap()
#             else:
#                 st.warning('Không có dữ liệu giá nhà.')
#
#         with st.expander('Tương quan biến phân loại và giá nhà', expanded=False):
#             def show_multiple_categorical_cols(col):
#                 subdf = df.groupby(col)['gia_nha'].mean().reset_index()
#                 index = np.argsort(subdf['gia_nha'])[::-1]
#                 col_name = subdf[col][index]
#                 col_value = subdf['gia_nha'][index]
#                 fig, ax = plt.subplots(figsize=(8, 3))
#                 ax.barh(
#                     col_name, col_value,
#                     color=plt.cm.Blues(col_value / col_value.max()),
#                     edgecolor='black',
#                     height=0.6
#                 )
#                 ax.set_title(f'Giá bán nhà trung bình theo {column_name[col]}')
#                 ax.set_xlabel('Giá nhà trung bình (triệu/m2)')
#                 ax.set_ylabel(f'{column_name[col]}')
#                 ax.grid(True, axis='x', alpha=0.3)
#                 st.pyplot(fig)
#
#             if 'gia_nha' in df.columns:
#                 for col in categorical_cols:
#                     show_multiple_categorical_cols(col)
#             else:
#                 st.warning('Không có dữ liệu giá nhà.')
#     else:
#         st.warning('Tệp rỗng hoặc không có dữ liệu.')
# else:
#     st.info('Hãy chọn một tệp CSV để bắt đầu.')
#
#


import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from unidecode import unidecode
from utils.preprocess import preprocess
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# file mapping tên cột
mapping_path = os.path.join(os.path.dirname(__file__), '..', 'datas', 'lookup', 'column_names.json')
with open(mapping_path, 'r', encoding='utf-8-sig') as f:
    column_name = json.load(f)

vietnam_geojson_path = os.path.join(os.path.dirname(__file__), '..', 'datas', 'lookup', 'district.geojson')
with open(vietnam_geojson_path, 'r', encoding='utf-8-sig') as f:
    vietnam_geojson = json.load(f)

st.set_page_config(layout='wide')
st.title('DỮ LIỆU THỐNG KÊ')
st.caption('Tải lên một tệp CSV để xem nhanh dữ liệu, thống kê mô tả và biểu đồ.')

uploaded = st.file_uploader('Chọn tệp CSV', type=['csv'])
if uploaded is not None and 'df' not in st.session_state:
    st.session_state['df'] = pd.read_csv(uploaded)

if 'df' in st.session_state:
    df = st.session_state['df']
    if df.shape[0] > 0:
        df = preprocess(df)

        # Xem nhanh dữ liệu
        st.subheader('Xem nhanh dữ liệu')
        df_display = df.copy()
        df_display = df_display.rename(columns=column_name)
        df_display = df_display.reset_index(drop=True)
        st.dataframe(df_display.head(20))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_display = [column_name[col] for col in numeric_cols]
        # Thống kê mô tả
        if numeric_cols:
            st.subheader('Thống kê mô tả')
            desc = df[numeric_cols].describe().T
            desc['median'] = df[numeric_cols].median()
            desc.index = numeric_cols_display
            st.dataframe(desc)
        else:
            st.info('Không phát hiện cột số để tính thống kê mô tả.')

        # ---------------- Biểu đồ đơn biến ----------------
        st.subheader('Biểu đồ đơn biến')

        # Histogram + Boxplot
        with st.expander('Biểu đồ cho cột số', expanded=False):
            for col in numeric_cols:
                fig = make_subplots(rows=1, cols=2, subplot_titles=[f'Histogram {column_name[col]}', f'Boxplot {column_name[col]}'])

                # Histogram
                fig.add_trace(go.Histogram(x=df[col], nbinsx=32, name='Histogram'), row=1, col=1)
                # Boxplot
                fig.add_trace(go.Box(y=df[col], name='Boxplot'), row=1, col=2)

                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)

        # Bar + Pie cho cột phân loại
        categorical_cols = ['huyen', 'loai_hinh_nha_o']
        with st.expander('Biểu đồ cho cột phân loại', expanded=False):
            for col in categorical_cols:
                counts = df[col].value_counts()
                labels = counts.index

                # Bar chart
                fig_bar = px.bar(x=labels, y=counts.values, labels={'x': column_name[col], 'y':'Số lượng'},
                                 title=f'Số lượng căn nhà theo {column_name[col]}', text=counts.values)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Pie chart
                fig_pie = px.pie(df, names=col, title=f'Tỉ lệ căn nhà theo {column_name[col]}')
                st.plotly_chart(fig_pie, use_container_width=True)

        # ---------------- Biểu đồ đa biến ----------------
        st.subheader('Biểu đồ đa biến')

        # Heatmap numeric correlations
        with st.expander('Tương quan biến số và giá nhà', expanded=False):
            if numeric_cols:
                corr = df[numeric_cols].corr()
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=[column_name[c] for c in numeric_cols],
                    y=[column_name[c] for c in numeric_cols],
                    colorscale='RdBu',
                    zmin=-1, zmax=1,
                    text=np.around(corr.values,2),
                    hoverinfo='text'
                ))
                fig_heatmap.update_layout(title='Mức độ tương quan các đặc trưng số', height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)

        # Categorical vs giá_nha
        with st.expander('Tương quan biến phân loại và giá nhà', expanded=False):
            if 'gia_nha' in df.columns:
                for col in categorical_cols:
                    subdf = df.groupby(col)['gia_nha'].mean().reset_index()
                    subdf = subdf.sort_values('gia_nha', ascending=False)
                    fig_cat = px.bar(subdf, x='gia_nha', y=col, orientation='h',
                                     labels={'gia_nha':'Giá nhà trung bình (triệu/m2)', col: column_name[col]},
                                     title=f'Giá bán nhà trung bình theo {column_name[col]}')
                    st.plotly_chart(fig_cat, use_container_width=True)

                # Bản đồ thực
                def normalize(df: pd.DataFrame):
                    df['huyen_std'] = df['huyen'].str.replace('Quận ', '')
                    df['huyen_std'] = df['huyen_std'].str.replace('Huyện ', '')
                    df['huyen_std'] = df['huyen_std'].apply(lambda x: unidecode(x.strip()).title())
                    df['city'] = 'Hanoi'
                    return df
                df = normalize(df)
                df_grouped = df.groupby('huyen_std')['gia_nha'].mean().reset_index()
                hanoi_geojson = {
                    "type": "FeatureCollection",
                    "features": [f for f in vietnam_geojson["features"] if f["properties"].get("Province") == "Ha Noi"]
                }

                width = 800
                height = width / 4 * 3

                fig = px.choropleth(
                    df_grouped,
                    geojson=hanoi_geojson,
                    locations='huyen_std',
                    featureidkey="properties.District",
                    color='gia_nha',
                    color_continuous_scale="Greens",
                    hover_name='huyen_std',
                    hover_data={'gia_nha': True},
                    width=width,
                    height=height
                )
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(title_text="Bản đồ nhiệt giá nhà trung bình theo quận/huyện tại Hà Nội")

                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('Tệp rỗng không có dữ liệu.')
else:
    st.info('Hãy chọn một tệp CSV để bắt đầu.')
