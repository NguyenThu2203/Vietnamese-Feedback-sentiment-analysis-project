import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from pyvi import ViTokenizer
import pickle

def preprocess_raw_input(raw_input, tokenizer):
    input_text_pre = list(tf.keras.preprocessing.text.text_to_word_sequence(raw_input))
    input_text_pre = " ".join(input_text_pre)
    input_text_pre_accent = ViTokenizer.tokenize(input_text_pre)
    tokenized_data_text = tokenizer.texts_to_sequences([input_text_pre_accent])
    vec_data = pad_sequences(tokenized_data_text, padding='post', maxlen=40)
    return vec_data

def inference_model(input_feature, model):
    output = model(input_feature).numpy()[0]
    result = output.argmax()
    conf = float(output.max())
    label_dict = {'Tiêu cực': 0, 'Tích cực': 1, 'Trung lập': 2}
    label = list(label_dict.keys())
    return label[int(result)], conf

def prediction(raw_input, tokenizer, model):
    input_model = preprocess_raw_input(raw_input, tokenizer)
    result, conf = inference_model(input_model, model)
    return result, conf

# Tải model và tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('model_cnn_bilstm.h5')
    with open("tokenizer_data.pkl", "rb") as input_file:
        tokenizer = pickle.load(input_file)
    return model, tokenizer

# Tải model và tokenizer
my_model, my_tokenizer = load_model_and_tokenizer()

# Giao diện Streamlit
st.title('Dự đoán Cảm xúc Văn bản Tiếng Việt')
st.write('Nhập văn bản của bạn vào ô dưới đây và nhận kết quả dự đoán cảm xúc.')

user_input = st.text_area('Nhập văn bản tại đây:')

if st.button('Dự đoán'):
    if user_input:
        result, conf = prediction(user_input, my_tokenizer, my_model)
        st.write(f'Kết quả: {result}')
        st.write(f'Độ tin cậy: {conf:.2f}')
    else:
        st.write('Vui lòng nhập văn bản để dự đoán.')
