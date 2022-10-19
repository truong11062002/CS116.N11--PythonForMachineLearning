from ast import operator
import streamlit as st 
import pandas as pd
import random
import numpy as np
from PIL import Image

st.markdown("""
# Họ và tên: Nguyễn Nhật Trường
# MSSV: 20522087
# Đây là bài tutorial
## 1. Giới thiệu streamlit
### 1.1 Giới thiệu chung
### 1.2 Cài đặt
## 2. Các thành phần cơ bản của giao dịch
""")

a_value = st.text_input("Nhập a: ")
b_value = st.text_input("Nhập b: ")

button = st.button("Calculate")
operator = st.radio("Chọn phép toán", ['+', '-','*', '/'])

image = Image.open("C:\CS116.N11\Assignments\BTTL_Streamlit\sunshine.png")
image_girl = Image.open("C:\CS116.N11\Assignments\BTTL_Streamlit\girl.jpg")
image_girl2 = Image.open("C:\CS116.N11\Assignments\BTTL_Streamlit\girl1.jpg")

st.image(image, caption='Sunrise by the mountains')
st.image(image_girl, caption="Girl xinh")

col1, col2, col3 = st.columns(3)

with col1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")
   st.header("A cow")
   st.image("https://static.streamlit.io/examples/owl.jpg")
   st.header("A owl")
   st.image("https://static.streamlit.io/examples/dog.jpg")

with col2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")
   st.header("A cow")
   st.image("https://static.streamlit.io/examples/owl.jpg")

with col3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")
   st.header("A cow")
   st.image("https://static.streamlit.io/examples/owl.jpg")
   st.header("A owl")


tab1, tab2, tab3 = st.tabs(["Girl1", "Girl2", "Girl3"])
with tab1:
   st.header("Girl1")
   st.image(image, width=400)

with tab2:
   st.header("Girl2")
   st.image(image_girl, width=400)

with tab3:
   st.header("Girl3")
   st.image(image_girl2, width=400)


uploaded_file = st.file_uploader("Chọn ảnh đẹp")
if uploaded_file is not None:
    # To read file as bytes to local disk:
    bytes_data = uploaded_file.getvalue()
    with open('image/' + uploaded_file.name, 'wb') as f:
        f.write(bytes_data)
    
    


# # accept many file
# uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
# for uploaded_file in uploaded_files:
#     bytes_data = uploaded_file.read()
#     st.write("filename:", uploaded_file.name)

#     with open('image/' + uploaded_file, 'wb') as f:
#         f.write(bytes_data)
#     # st.write(bytes_data)


if button:
    if operator == '+':
        st.text_input("Kết quả: ", float(a_value) + float(b_value))
    
    if operator == '-':
        st.text_input("Kết quả: ", float(a_value) - float(b_value))
    if operator == '*':
        st.text_input("Kết quả: ", float(a_value) * float(b_value))
    if operator == '/':
        st.text_input("Kết quả: ", float(a_value) / float(b_value))

#     st.text_input("Kết quả: ", float(a_value) + float(b_value))


df = pd.DataFrame(
    np.random.randn(10,5),
    columns=('col %d' % i for i in range(5))
)

st.table(df)
st.line_chart(df)
st.area_chart(df)
st.bar_chart(df)
