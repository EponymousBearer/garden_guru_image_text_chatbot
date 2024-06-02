import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from PIL import Image
# import google.generativeai as genai
from transformers import pipeline

def generate_response(predictions):
    """
    Generate a response from the chatbot based on the classification results.

    Args:
        predictions (list): A list of dictionaries containing the classification results.

    Returns:
        str: A response string like "The plant is a [label] with [score]% confidence."
    """
    response = ""
    for p in predictions:
        response += f"The plant is a {p['label']} with {round(p['score'] * 100, 1)}% confidence.\n"
    return response

# Load environment variables
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("api_key")

# Set up Google Gemini-Pro AI model
# genai.configure(api_key=GOOGLE_API_KEY)

def load_huggingface_text_model(model_name):
    model = pipeline('text-generation', model=model_name)
    return model

# load gemini-pro model
# def gemini_pro():
#     model = genai.GenerativeModel('gemini-pro')
#     return model

image_captioning_model = pipeline("image-classification", model="dima806/medicinal_plants_image_detection")

def huggingface_image_caption(model, prompt, image):
    image_input = open(image, "rb").read()
    image_features = model(images=image_input)[0][0, 0, :]
    text_input = model.tokenizer(prompt, return_tensors="pt").input_ids
    text_features = model(text_input)[0][0, :, :]
    similarity = model.compute_similarity(image_features, text_features)
    most_similar_index = similarity.argmax().item()
    return model.tokenizer.decode(text_input[0][most_similar_index])

# Load gemini vision model
# def gemini_vision():
#     model = genai.GenerativeModel('gemini-pro-vision')
#     return model

# get response from gemini pro vision model
# def gemini_visoin_response(model, prompt, image):
#     response = model.generate_content([prompt, image])
#     return response.text

# Set page title and icon

st.set_page_config(
    page_title="Chat With Gemi",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    user_picked = option_menu(
        "Google Gemini AI",
        ["ChatBot", "Plant Identification"],
        menu_icon="robot",
        icons = ["chat-dots-fill", "image-fill"],
        default_index=0
    )

def roleForStreamlit(user_role):
    if user_role == 'model':
        return 'assistant'
    else:
        return user_role
    

# if user_picked == 'ChatBot':
#     model = gemini_pro()
    
#     if "chat_history" not in st.session_state:
#         st.session_state['chat_history'] = model.start_chat(history=[])

#     st.title("🤖TalkBot")

#     #Display the chat history
#     for message in st.session_state.chat_history.history:
#         with st.chat_message(roleForStreamlit(message.role)):    
#             st.markdown(message.parts[0].text)

#     # Get user input
#     user_input = st.chat_input("Message TalkBot:")
#     if user_input:
#         st.chat_message("user").markdown(user_input)
#         reponse = st.session_state.chat_history.send_message(user_input)
#         with st.chat_message("assistant"):
#             st.markdown(reponse.text)

if user_picked == 'ChatBot':
    model = load_huggingface_text_model('')
    
    if "chat_history" not in st.session_state:
        st.session_state['chat_history'] = []

    st.title("🤖TalkBot")

    # Display the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(roleForStreamlit(message.role)):    
            st.markdown(message.parts[0].text)

    # Get user input
    user_input = st.chat_input("Message TalkBot:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "parts": [{"text": user_input}]})
        response = model(user_input, max_length=100)
        st.session_state.chat_history.append({"role": "assistant", "parts": [{"text": response[0]['generated_text']}]})
        
if user_picked == 'Plant Identification':
    # model = image_captioning_model
    
    file_name = st.file_uploader("Upload a plant image")

    if file_name is not None:
        # Display the uploaded image
        image = Image.open(file_name)
        st.image(image, use_column_width=True)

        # Classify the image using the Hugging Face model
        predictions = image_captioning_model(image)

        # Display the classification results
        st.header("Classification Results")
        for p in predictions:
            st.subheader(f"{p['label']}: {round(p['score'] * 100, 1)}%")

        # Generate a response from the chatbot
        response = generate_response(predictions)
        st.write(response)

    st.title("Plant Identification")

    image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    user_prompt = st.text_input("Enter the prompt for Plant Identification:")

    if st.button("Generate Caption"):
        caption_response = huggingface_image_caption(model, user_prompt, image)
        st.info(caption_response)
        
# if user_picked == 'Image Captioning':
#     model = image_captioning_model

#     st.title("🖼️Image Captioning")

#     image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

#     user_prompt = st.text_input("Enter the prompt for image captioning:")

#     if st.button("Generate Caption"):
#         load_image = Image.open(image)

#         colLeft, colRight = st.columns(2)

#         with colLeft:
#             st.image(load_image.resize((800, 500)))

#         caption_response = gemini_visoin_response(model, user_prompt, load_image)

#         with colRight:
#             st.info(caption_response)
