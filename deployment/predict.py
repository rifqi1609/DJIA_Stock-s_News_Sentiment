# Import Libraries
import streamlit as st
import pandas as pd                                       # For Preprocessing
import numpy as np
import tensorflow as tf
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow_hub as tf_hub
from tensorflow.keras.models import load_model
import string

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Function for Text Preprocessing
def text_preprocessing(text):
  # Case folding
  text = text.lower()

  # Newline removal (\n)
  text = re.sub(r"\\n", " ",text)

  # Whitespace removal
  text = text.strip()

  # Puntuaction
  text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub(r'\d+', '', text)

  # Non-letter removal (such as emoticon, symbol (like μ, $, 兀), etc
  text = re.sub("[^A-Za-z\s']", " ", text)

  # Tokenization
  tokens = word_tokenize(text)

  # Define Stopwords
  stpwds_id = list(set(stopwords.words('english')))

  # Stopwords removal
  tokens = [word for word in tokens if word not in stpwds_id]

  # Lemmatizer  
  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(word) for word in tokens]

  # Combining Tokens
  text = ' '.join(tokens)
  return text

# Preprocessing layer
def preprocessing_layer(text_tensor):
    def _inner_py_func(text_b):
        text = text_b.decode('utf-8')
        return text_preprocessing(text)
    
    result = tf.numpy_function(_inner_py_func, [text_tensor], tf.string)
    return result

# Multiple Inferences
model_path = 'deployment/final_model.keras'
model = tf.keras.models.load_model(
    model_path, 
    custom_objects={
        'preprocessing_layer': preprocessing_layer,
        'KerasLayer': tf_hub.KerasLayer
    },
    safe_mode=False
)

labels = ['Negative', 'Neutral', 'Positive'] 

def run():
    st.title('News Prediction')
    st.markdown('## Input Single Data')
   
    # Form
    with st.form('form_input'):
        new_data = st.text_input('Input Headline and Summary of News')
        submit_btn = st.form_submit_button('Predict')

    if submit_btn:
        data_inf = new_data
        sample_processed = text_preprocessing(data_inf)
        result = model.predict(tf.constant([sample_processed]))

        # Transfrom to Target Label
        higher_value = np.argmax(result[0])
        sentiment_result = labels[higher_value]
        st.write(f'The sentiment prediction of this news is "{sentiment_result}"')

if __name__ == '__main__':
    run()


def run_file():
    st.markdown('## Input Multiple Data')

    # 2. Upload File CSV
    uploaded_file = st.file_uploader("Upload Data", type=["csv"])

    if uploaded_file is not None:
        # Load New Data
        df_inf = pd.read_csv(uploaded_file)

        # Prediction Button
        if st.button('Predict All'):
            try:
                predictions = model.predict(np.array(df_inf))
                target_indices = np.argmax(predictions, axis=1)

                predicted_labels = [labels[idx] for idx in target_indices]

                df_result = pd.DataFrame({
                    'text': df_inf,
                    'label': predicted_labels
                })
                
                st.success("Prediction Complete!")
                st.dataframe(df_inf)

                # Tombol Download Hasil
                csv = df_inf.to_csv(index=False).encode('utf-8')
                st.download_button("Download Prediction", csv, "prediction.csv", "text/csv")
            
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == '__main__':
    run_file()

