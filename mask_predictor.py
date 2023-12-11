import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import binary_crossentropy
import cv2

def dice_metric(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    
    return (numerator + 1) / (denominator + 1)

def combined_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + 0.5 * (1 - dice_metric(y_true, y_pred))

class MaskPredictor:

    
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'combined_loss': combined_loss, 'dice_metric': dice_metric})

        
    def preprocess_image(self, image_path, target_size=(448, 768)):
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image

    def predict_mask(self, image):
        predicted_mask = self.model.predict(image)
        return predicted_mask

    def post_process_mask(self, predicted_mask, threshold=0.5, kernel_size=(20, 20)):
        mask_for_morphology = np.squeeze(predicted_mask[0, ..., 0])

        # Add Gaussian blur before morphological operations
        mask_for_morphology = cv2.GaussianBlur(mask_for_morphology, (25, 25), 0)

        # Define a kernel for morphological operations
        kernel = np.ones(kernel_size, np.uint8)

        # Perform erosion and dilation
        eroded_mask = cv2.erode(mask_for_morphology, kernel, iterations=1)
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

        # Apply threshold after morphological operations
        # dilated_mask[dilated_mask > threshold] = 1
        # dilated_mask[dilated_mask <= threshold] = 0

        cv2.imwrite("dilated_mask.png", dilated_mask * 255)

        return dilated_mask

    def apply_mask_to_image(self, image, mask):
        masked_image = image[0] * np.expand_dims(mask, axis=-1)
        return masked_image

    def process_image(self, image_path, threshold=0.5):
        image = self.preprocess_image(image_path)
        predicted_mask = self.predict_mask(image)
        processed_mask = self.post_process_mask(predicted_mask, threshold)
        masked_image = self.apply_mask_to_image(image, processed_mask)
        return masked_image
