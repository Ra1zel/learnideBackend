from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


def vitBase_pred(image: Image):
    processor = ViTImageProcessor.from_pretrained(
        'google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224')
    inputs = processor(images=image, return_tensors="pt")
    return "Hello"
    outputs = model(**inputs)
    logits = outputs.logits

    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]
