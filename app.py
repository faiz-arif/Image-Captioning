import streamlit as st
from utils import save_uploaded_image
from image_prediction import ImagePrediction

model = ImagePrediction()

def main():
    st.title("Image Captioning")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        image_path = save_uploaded_image(uploaded_file)
        predictions = model.predict_step([image_path])

        st.subheader("Predictions:")
        for i, pred in enumerate(predictions):
            st.write(f"{i + 1}. {pred}")

if __name__ == "__main__":
    main()