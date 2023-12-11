import cv2
import numpy as np

def preprocess(image):
    
    def remove_noise(image):
        """
            Try to remove noise from the image

            :param image: Image to remove noise from (GRAYSCALE)

            :return: Image with noise removed (GRAYSCALE)
        """
        horizontal_kernel = (3, 1)

        for _ in range(2):
            image = cv2.GaussianBlur(image, horizontal_kernel, 1)
            _, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)


        square_kernel = np.ones((2, 2))

        image = cv2.erode(image, square_kernel, iterations=2)
        image = cv2.dilate(image, square_kernel, iterations=3)

        return image
    
    image_filtered = image

    image_filtered_grayscale = cv2.cvtColor(image_filtered, cv2.COLOR_RGB2GRAY)
    image_filtered_grayscale = cv2.convertScaleAbs(image_filtered_grayscale)

    # Binarize the image using adaptive thresholding to keep a maximum of information
    image_thresholded = cv2.adaptiveThreshold(image_filtered_grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)

    # Remove noise from the image, like the grass' shadows
    image_denoised = remove_noise(image_thresholded)

    return image_denoised
