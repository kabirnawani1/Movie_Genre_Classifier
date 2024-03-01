import streamlit as st
import model

# Set up Streamlit app title
st.title('Movie Genre Classifier')
st.title('Enter Plot')

# Function to load the model and tokenizer
@st.cache_resource()
def load_model():
    return model.load_model_and_tokenizer()

# Load model, device, tokenizer, and max length
loaded_model, device, tokenizer, MAX_LEN = load_model()

# Text area to input the plot
text = st.text_area("Plot")
product = None
label = 'Submit'

# Button to trigger prediction
if st.button(label):
    # Check if text area is not empty
    if text != '':
        product = model.predict_genre(
            text, loaded_model, device, tokenizer, MAX_LEN)
        product = ', '.join(product)

    # Display prediction
    if product is not None:
        data_load_state = st.text('The genre is')
        st.write(product)
