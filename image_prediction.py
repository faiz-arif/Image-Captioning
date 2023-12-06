from transformers import VisionEncoderDecoderModel, AutoTokenizer,ViTImageProcessor
import torch
from PIL import Image

class ImagePrediction:
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
    def predict_step(self, image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = ImagePrediction.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(ImagePrediction.device)

        output_ids = ImagePrediction.model.generate(pixel_values, **ImagePrediction.gen_kwargs)

        preds = ImagePrediction.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds




