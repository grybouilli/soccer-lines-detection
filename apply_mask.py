from mask_predictor import MaskPredictor
import matplotlib.pyplot as plt

# Chemin vers le modèle et l'image
MODEL_PATH = "model_light_1_tr05.h5"
IMAGE_PATH = "videos/log2/029-rgb.png"

# Initialiser le prédicteur de masque
mask_predictor = MaskPredictor(MODEL_PATH)

# Traiter l'image et obtenir le masque
masked_image = mask_predictor.process_image(IMAGE_PATH)

# Afficher l'image masquée
plt.imshow(masked_image)
plt.show()

# Enregistrer l'image masquée
plt.imsave('masked_image.png', masked_image)
