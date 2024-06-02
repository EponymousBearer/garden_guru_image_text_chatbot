import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from PIL import Image
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.pipelines import TextGenerationPipeline

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

def load_huggingface_text_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    return pipeline

def gemini_pro():
    model = load_huggingface_text_model('NousResearch/Llama-2-7b-chat-hf')
    return model

def load_huggingface_text_model(model_name):
    model = pipeline('text-generation', model=model_name)
    return model

image_captioning_model = pipeline("image-classification", model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")

def huggingface_image_caption(model, prompt, image):
    image_input = open(image, "rb").read()
    image_features = model(images=image_input)[0][0, 0, :]
    text_input = model.tokenizer(prompt, return_tensors="pt").input_ids
    text_features = model(text_input)[0][0, :, :]
    similarity = model.compute_similarity(image_features, text_features)
    most_similar_index = similarity.argmax().item()
    return model.tokenizer.decode(text_input[0][most_similar_index])

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
    
if user_picked == 'ChatBot':
    model = gemini_pro()
    
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
    st.title("Plant Identification")
    
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