import streamlit as st
import os
from utils.caption_generator import display_caption, generate_caption

st.title("Image Captioning App")
st.write("Upload an image to generate a caption.")

image_path = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image_path is not None:
    # Save the uploaded image temporarily
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(image_path.read())
    
    # Define model paths
    model_path = "model.keras"
    tokenizer_path = "tokenizer.pkl"
    feature_extractor_path = "feature_extractor.keras"
    
    # Generate caption
    caption = generate_caption(temp_image_path, model_path, tokenizer_path, feature_extractor_path)
    
    # Create a centered layout for the caption
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <h3><b>{caption}</b></h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create a centered layout for the image
    col1, col2, col3 = st.columns([1, 6, 1])  # Adjust the middle column width as needed
    with col2:
        st.image(temp_image_path, use_column_width=True)  # Let Streamlit handle the width
    
    # Clean up the temporary file
    os.remove(temp_image_path)