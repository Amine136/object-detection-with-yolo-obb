import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np
import math


def preprocess_image(image: Image) -> np.ndarray:
    # Redimensionner l'image à 640x640, taille par défaut de YOLOv8
    image = image.resize((640, 640))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# Charger le modèle YOLO OBB
model = YOLO('model.pt')  # Remplacez 'model_obb.pt' par le chemin vers votre modèle OBB

st.title("YOLOv8 OBB")

# Chargement de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger et afficher l'image originale
    image = Image.open(uploaded_file)
    st.image(image, caption="Image originale", use_column_width=True)
    
    # Exécuter le modèle pour obtenir les détections avec NMS et seuil de confiance
    results = model.predict(source=image, imgsz=640, conf=0.5)  # Ajuster le seuil de confiance si nécessaire

    # Créer une copie de l'image pour dessiner les boîtes orientées
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
 
    
    # Boucle sur les détections pour dessiner les boîtes orientées sans étiquettes
    for box in results[0].obb:
        # Récupérer les coordonnées et l'angle de rotation pour la boîte orientée
        x_center, y_center, width, height, angle = box.xywhr.tolist()[0]
        # Calculer les coins du rectangle orienté
        angle_rad = angle
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Coordonnées des coins
        corners = [
            (x_center + width / 2 * cos_a - height / 2 * sin_a, y_center + width / 2 * sin_a + height / 2 * cos_a),
            (x_center - width / 2 * cos_a - height / 2 * sin_a, y_center - width / 2 * sin_a + height / 2 * cos_a),
            (x_center - width / 2 * cos_a + height / 2 * sin_a, y_center - width / 2 * sin_a - height / 2 * cos_a),
            (x_center + width / 2 * cos_a + height / 2 * sin_a, y_center + width / 2 * sin_a - height / 2 * cos_a),
        ]
        
        # Dessiner le rectangle orienté
        draw.polygon(corners, outline="red", width=3)
    
    # Afficher l'image annotée et le nombre de détections
    st.image(annotated_image, caption="Image annotée (boîtes orientées)", use_column_width=True)
    st.write(f"Nombre de détections : {len(results[0].obb)}")
