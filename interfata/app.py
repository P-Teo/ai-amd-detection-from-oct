import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os
from PIL import Image

# Importăm funcțiile tale din fișierele separate
from processing.process_t1 import process_image as process_v1
from processing.process_t2 import process_oct_image as process_v2


import torch
import torch.nn as nn
from torchvision import models
import streamlit as st



from unet_model import UNet

from tensorflow.keras.models import load_model as tf_load_model

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_h5_model(path):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model

    try:
        # 1. Creăm baza cu greutățile ImageNet (pentru a umple cele 103 straturi)
        inputs = Input(shape=(224, 224, 3))
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
        # 2. Reconstruim structura finală
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # Numele 'dense_6' din eroarea ta anterioară
        outputs = Dense(4, activation='softmax', name='dense_6')(x) 
        model = Model(inputs=inputs, outputs=outputs)
        # 3. Încărcăm DOAR cele 2 straturi pe care le ai în fișier
        # skip_mismatch=True va ignora straturile MobileNet și va încărca doar ce găsește
        # by_name=True va căuta stratul 'dense_6' în fișierul tău
        model.load_weights(path, by_name=True, skip_mismatch=True)
        return model
    except Exception as e:
        st.error(f"Eroare la asamblarea hibridă: {e}")
        return None



 
def preprocess_for_h5(image_path):
    # Citire imagine
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (224, 224))
    image = cv2.medianBlur(image, 3)
    # 3. CLAHE (Contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    # 4. Normalize 0-1
    image = image.astype("float32") / 255.0
    # 5. Convert to 3 channels (Stack)
    image_3ch = np.stack([image, image, image], axis=-1)
    return np.expand_dims(image_3ch, axis=0)

   

@st.cache_resource
def load_model_unet(path):
    device = torch.device("cpu")
    model = UNet(n_classes=5)  
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


@st.cache_resource
def load_model(path):
    device = torch.device("cpu")
    #  model este nn.Module, NU dict
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model

import torch
from torchvision.models.resnet import ResNet

def load_model_d(path):
    device = torch.device("cpu")
    torch.serialization.add_safe_globals([ResNet])
    # Încarcă modelul complet (weights_only=False pentru modele salvate complet)
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model



# Configurare pagină
st.set_page_config(page_title="OCT Analysis Multi-Process", layout="wide")



# Încărcăm modelele 
model_t1 = load_model("models/segmentare/verificare_segmentare/oct_modelvar1_2_resnet18.pth")
model_t2 = load_model("models/segmentare/verificare_segmentare/oct_modelvar2_2_resnet18.pth")
model_seg = load_model_unet("models/segmentare/construire_segmentare/OIMHS_segmentare_unet_oct.pth")
model_seg_class = load_model("models/segmentare/verificare_segmentare/oct_oimhs_resnet18.pth")
model_pth_direct = load_model_d("models/nu_segmentare/oct_cnn_model_ResNet18.pth") 
model_h5_direct = load_h5_model("models/nu_segmentare/oct_cnn_model_MobileNetV2.h5")


st.title("Sistem de Analiză OCT")
st.sidebar.header("Configurare")

option = st.sidebar.selectbox(
    "Alege fluxul de lucru:",
    ("Procesare Tip 1 (process_t1)",
     "Procesare Tip 2 (process_t2)",
     "Segmentare cu model direct",
     "Model Direct (.pth)",
     "Model Direct (.h5)")
)

uploaded_file = st.sidebar.file_uploader("Încarcă imaginea...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagine Originală")
        st.image(uploaded_file, use_container_width=True)
    with col2:
        st.subheader("Rezultat Analiză")

         # Dicționar pentru fluxurile de segmentare (T1, T2, UNet) - au doar 3 clase
        clase_3 = {0: "CNV", 1: "DRUSEN", 2: "NORMAL"}
        # Dicționar pentru modelele directe (.pth și .h5) - au 4 clase din antrenament
        clase_4 = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
        clase_dict_pytorch = {0: "CNV",  1: "DRUSEN", 2: "NORMAL"}
        clase_dict_h5 = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}  

        prediction = None

        # --- Logica fluxului ---
        if option == "Procesare Tip 1 (process_t1)":
            _, mask, overlay = process_v1(temp_path)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            input_tensor = torch.from_numpy(cv2.resize(mask, (224,224))).permute(2,0,1).float().unsqueeze(0)/255.0
            prediction = model_t1(input_tensor)

        elif option == "Procesare Tip 2 (process_t2)":
            _, mask, overlay = process_v2(temp_path)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            input_tensor = torch.from_numpy(cv2.resize(mask, (224,224))).permute(2,0,1).float().unsqueeze(0)/255.0
            prediction = model_t2(input_tensor)

        elif option == "Segmentare cu model direct":
            # --- 1. PREPROCESARE  ---
            img_orig = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            
            #  rezoluția de antrenare UNet
            TARGET_SIZE = (512, 512)
            img_res = cv2.resize(img_orig, TARGET_SIZE)
            
            # CLAHE 
            mask_nonblack = img_res >20
            clahe = cv2.createCLAHE(clipLimit=1.9, tileGridSize=(9,10))
            
            processed_img = img_res.copy()
            clahe_applied = clahe.apply(processed_img)
            processed_img[mask_nonblack] = clahe_applied[mask_nonblack]
            processed_img[processed_img <= 0] = 0 
            
            # --- 2. INFERENȚĂ UNet ---
            # Normalizare la [0, 1] și conversie în Tensor
            img_input = processed_img.astype(np.float32) / 255.0
            img_t = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).to(device)
            
            model_seg.eval()
            with torch.no_grad():
                logits = model_seg(img_t)
                # masca 
                pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

            # --- 3. GENERARE MASCĂ COLOR ---
            # Folosim culorile din cod
            mask_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
            mask_rgb[pred == 1] = [255, 0, 0]   # Roșu
            mask_rgb[pred == 2] = [0, 255, 0]   # Verde 
            mask_rgb[pred == 3] = [0, 0, 255]   # Albastru
            mask_rgb[pred == 4] = [255, 255, 0] # Galben

            # Afișez rezultatul în aplicație
            st.image(mask_rgb, caption="Masca Segmentată (Trebuie să fie densă ca în exemplul tău)", use_container_width=True)

            # --- 4. CLASIFICARE (ResNet18) ---
            # ResNet a învățat pe 160x160
            mask_for_cls = cv2.resize(mask_rgb, (160, 160))
            input_cls = torch.from_numpy(mask_for_cls).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad():
                prediction = model_seg_class(input_cls)

        elif option == "Model Direct (.pth)":
            st.info("Analiză directă pe imaginea brută (ResNet18)")
            img_bgr = cv2.imread(temp_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Resize (224, 224) + Normalizare ImageNet
            img_res = cv2.resize(img_rgb, (224, 224))
            input_tensor = torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0
            
            #  mediile și deviațiile standard standard folosite la antrenament
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            input_tensor = (input_tensor - mean) / std
            input_tensor = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model_pth_direct(input_tensor)


        elif option == "Model Direct (.h5)":
            st.info("Analiză directă MobileNetV2 (Preprocesare Custom)")
            if model_h5_direct is None:
                st.error("Modelul H5 nu a putut fi încărcat!")
            else:
                #  PREPROCESARE
                img_h5 = preprocess_for_h5(temp_path)
                
                if img_h5 is not None:
                    preds_h5 = model_h5_direct.predict(img_h5)
                    clasa_idx = np.argmax(preds_h5)
                    
                    # Debug: scoruri pentru fiecare clasă
                    clase_h5 = {0: "CNV", 1: "CNV", 2: "DRUSEN", 3: "NORMAL"}
                    
                    st.write(f"Încredere: {np.max(preds_h5)*100:.2f}%")
                    st.success(f"**Diagnostic AI (.h5):** {clase_h5.get(clasa_idx)}")
                    
                    #  probabilitățile sub formă de tabel
                    import pandas as pd
                    prob_df = pd.DataFrame(preds_h5, columns=clase_h5.values())
                    st.bar_chart(prob_df.T)

        # --- Afișare rezultat PyTorch ---
        if prediction is not None:
            with torch.no_grad():
                idx = torch.argmax(prediction, dim=1).item()
                
                # Verific model - pentru a alege dicționarul corect
                if option in ["Model Direct (.pth)"]:
                    nume_diagnostic = clase_4.get(idx, "NEC")
                else:
                    nume_diagnostic = clase_3.get(idx, "NEC")
                
                st.success(f"**Diagnostic AI:** {nume_diagnostic}")

    os.unlink(temp_path)
else:
    st.info("Aștept încărcarea unei imagini din meniul lateral.")