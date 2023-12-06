from PIL import Image

def save_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    temp_path = "temp/temp_image.jpg"
    image.save(temp_path)
    return temp_path