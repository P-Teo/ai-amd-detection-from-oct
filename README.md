[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/BzgEFjMi)

# Clasificare AMD (Degenerescența Maculară Legată de Vârstă) 

Acest proiect investighează clasificarea imaginilor OCT pentru **Degenerescența Maculară Legată de Vârstă (AMD)**, folosind **rețele neuronale convoluționale (CNN)**, atât **fără segmentare**, cât și **cu segmentare explicită** a structurilor retiniene.

---

## 📚 Clase utilizate

- **CNV** – Choroidal Neovascularization  
- **DRUSEN** – Depuneri de drusen  
- **NORMAL** – Retină sănătoasă  

---

## 📂 Dataset-uri utilizate

### Dataset principal
- **Kermany OCT 2018**  
  https://www.kaggle.com/datasets/paultimothymooney/kermany2018  

### Dataset extern
- **NEH UT OCT Dataset**  
  https://www.kaggle.com/datasets/orvile/neh-ut-oct-dataset  

---

## 🔁 Fluxul proiectului

<img width="3240" height="1630" alt="image" src="https://github.com/user-attachments/assets/92389eb9-7fa8-4e9a-b769-ddbf516dc06d" />



### 🧪 Faza 1 – Clasificare fără segmentare
1. **T1** – Testarea unui model CNN existent  
   https://github.com/ACM-Research/image-classification-cnn/blob/master/OCT-cnn/python_model/variables/model.h5
2. **T2 – T3** – Antrenarea a două modele CNN pe dataset-ul Kermany2018
3. **T4 – T5** – Evaluarea fiecărui model:
   - pe setul de test din același dataset
   - pe un dataset OCT extern
4. Compararea performanței pe:
   - acuratețe globală
   - performanță pe clasă
   - capacitate de generalizare

     
### 🧩 Faza 2 – Clasificare cu segmentare explicită
În cadrul proiectului, am testat trei abordări de segmentare pentru a îmbunătăți clasificarea AMD:

Tip 1 – Segmentare custom (V1)
Am realizat o procesare inițială a imaginilor pentru segmentarea straturilor retiniene folosind intensitatea pixelilor și filtrări simple, generând măști color. Pe aceste măști am antrenat ulterior un model de clasificare pentru a evalua performanța.

 Tip 2 – Segmentare custom (V2 – îmbunătățită)
Am extins procesarea din V1 cu corectări suplimentare și umpleri inteligente între straturi, obținând măști mai curate și consistente. Modelul de clasificare a fost antrenat pe aceste măști pentru a compara îmbunătățirea performanței față de V1.

Tip 3 – Segmentare automată + clasificare (OIMHS)
Am antrenat mai întâi un model de segmentare (U-Net) pe un dataset extern cu măști manuale (https://springernature.figshare.com/articles/dataset/OIMHS_dataset/23508453?file=42522673). Apoi, acest model a fost folosit pentru a genera măști pe propriul nostru dataset. În final, am antrenat modelul de clasificare pe imaginile segmentate automat pentru validarea performanței pe clasele CNV, DRUSEN și NORMAL.


### 💻 Faza 3 - Interfață Streamlit pentru analiza OCT

Am creat o interfață interactivă cu **Streamlit** care permite rularea rapidă a fluxurilor de segmentare și clasificare pe imaginile OCT, pentru a verifica ușor performanța modelelor și a vizualiza rezultatele.

---

## 🧪 Faza 1 – Clasificare fără segmentare
**Antrenare directă pe imagini OCT brute**

---

## 🔹 Modele utilizate

### Model 1 – MobileNetV2 (TensorFlow / Keras)
- **Preprocesare:** grayscale, 224×224, Median Blur, CLAHE, normalizare [0,1], conversie la 3 canale
- **Arhitectură:** MobileNetV2 (ImageNet) + GlobalAveragePooling + Dense (4 clase (se elimină dme după), Softmax)
- **Antrenare:** Adam, Sparse Categorical Crossentropy, 10 epoci

---

### Model 2 – ResNet18 (PyTorch)
- **Preprocesare:** redimensionare 224×224, flip orizontal aleator, normalizare ImageNet
- **Arhitectură:** ResNet18 (ImageNet) cu strat final adaptat la clase
- **Antrenare:** SGD cu momentum, CrossEntropyLoss, 10 epoci

---

## 📊 Rezultate – Fără segmentare

### 🔸 Set de test (Kermany2018)

| Model | Acuratețe | CNV | DRUSEN | NORMAL |
|------|----------|-----|--------|--------|
| MobileNetV2 | ~93.9% | 99% | 82% | 98% |
| ResNet18 | ~98.7% | 100% | 96% | 100% |
| Model GitHub | ~97.66% | 97% | 95% | 99% |

| MobileNetV2 | ResNet18 |
|-------------|----------|
| <img width="320" src="https://github.com/user-attachments/assets/91c08296-59e8-455c-aba0-3b5b4858d1ae" /> | <img width="320" src="https://github.com/user-attachments/assets/8ce73d49-c840-4bf8-9334-0df161ee51b7" /> |



---

### 🔸 Dataset extern

| Model | Acuratețe | CNV | DRUSEN | NORMAL |
|------|----------|-----|--------|--------|
| MobileNetV2 | ~58.46% | 76% | 11% | 96% |
| ResNet18 | ~74.06% | 93% | 45% | 96% |
| Model GitHub | ~43% | - | - | - |

| MobileNetV2 | ResNet18 |
|-------------|----------|
| <img width="320" height="220" alt="MobileNetV2" src="https://github.com/user-attachments/assets/bbf689a6-c9af-4b85-a15f-189d447147a8" /> | <img width="320" height="220" alt="ResNet18" src="https://github.com/user-attachments/assets/41cf658d-bfbf-4ad1-91cd-71e1b99bcc94" /> |


---

## 🖋️ Concluzii – Fără segmentare

- **MobileNetV2**
  - Performanță bună pe dataset-ul intern
  - Scădere semnificativă pe dataset extern

- **ResNet18**
  - Performanță superioară pe ambele dataset-uri
  - Capacitate foarte bună de generalizare

---

## 🧩 Faza 2 – Clasificare cu segmentare explicită

După clasificarea fără segmentare, proiectul a fost extins prin introducerea **segmentării explicite a imaginilor OCT**, urmată de **antrenarea modelelor de clasificare pe imaginile segmentate**.

| Model  | CNV | DRUSEN | NORMAL |
|--------|-----|--------|--------|
| ORIGINAL  | <img width="220" alt="ORIGINAL CNV" src="https://github.com/user-attachments/assets/224dffe2-f76d-40dc-af2e-1804919e3d8c" /> | <img width="220" alt="ORIGINAL DRUSEN" src="https://github.com/user-attachments/assets/bdfa32cf-fa1b-49b3-86be-3415aeff0d84" /> | <img width="220" alt="ORIGINAL NORMAL" src="https://github.com/user-attachments/assets/674bc761-dbea-48f2-a01e-c3c64a13cca8" /> |
| V1  | <img width="220" alt="V1 CNV" src="https://github.com/user-attachments/assets/0507229e-c233-482b-8f2c-4f5e494d507b" /> | <img width="220" alt="V1 DRUSEN" src="https://github.com/user-attachments/assets/6ff046e0-89a6-4bc9-b545-8e7cf004bd99" /> | <img width="220" alt="V1 NORMAL" src="https://github.com/user-attachments/assets/0e4f0650-336c-4e30-a843-716e6f6da962" /> |
| V2  | <img width="220" alt="V2 CNV" src="https://github.com/user-attachments/assets/6574818c-5399-4999-8fa0-f1ec664fb980" /> | <img width="220" alt="V2 DRUSEN" src="https://github.com/user-attachments/assets/7ccc7952-4129-4ecd-ac5d-8288768106ae" /> | <img width="220" alt="V2 NORMAL" src="https://github.com/user-attachments/assets/caa41f7c-663b-43ed-8dc0-631b0751d0b3" /> |
| OIMHS  | <img width="220" alt="OIMHS CNV" src="https://github.com/user-attachments/assets/17cc0e89-3491-4279-a67c-a4eaf7c8c7e5" /> | <img width="220" alt="OIMHS DRUSEN" src="https://github.com/user-attachments/assets/52fb03f6-2462-4664-922c-9e39da7f535a" /> | <img width="220" alt="OIMHS NORMAL" src="https://github.com/user-attachments/assets/dd312d80-1693-404b-9d6d-a9a3dbb427e7" /> |



---

## 🔹 Tipuri de segmentare testate

### 🟢 Tip 1 – Segmentare custom (V1)
- Bazată pe intensitatea pixelilor, CLAHE și filtrare pe coloană
- Generare mască color (verde, roșu, albastru, roz)
- Post-procesare: dilatări morfologice, Median Blur, Gaussian Blur

---

### 🟢 Tip 2 – Segmentare custom (V2 – îmbunătățită)
- Umplere inteligentă între straturi
- Extindere controlată a zonelor albastre și roz
- Corectare artefacte și măști mai consistente structural

---

### 🟢 Tip 3 – Segmentare automată + clasificare

- **Segmentare:** U-Net custom (PyTorch)  
  - Input: OCT grayscale (1 canal)  
  - Output: mască semantică (5 clase)  
  - Loss: CrossEntropy + Dice  
  - Adam, LR=1e-4, batch=4, 30 epoci  

- **Clasificare:** ResNet18 (ImageNet)  
  - Input: imagini segmentate automat  
  - Output: 3 clase (CNV, DRUSEN, NORMAL)  
  - Mixed Precision Training, Adam


---

## 📊 Rezultate – Fără segmentare

### 🔸 Set de test (Kermany2018)

| Model | Acuratețe | CNV | DRUSEN | NORMAL |
|------|----------|-----|--------|--------|
| V1 | ~94.75% | 95% | 97% | 92% |
| V2 | ~92.56% | 97% | 90% | 90% |
| OIMHS | ~92.84% | 98% | 91% | 89% |


| V1 | V2 | OIMHS |
|----|----|-------|
| <img width="320" height="220" alt="V1" src="https://github.com/user-attachments/assets/f89e2c97-fc78-493c-8881-4dfa4f05f235" /> | <img width="320" height="220" alt="V2" src="https://github.com/user-attachments/assets/cdf35def-14b9-4f35-addf-bbba795ec61f" /> | <img width="320" height="220" alt="OIMHS" src="https://github.com/user-attachments/assets/d7c9c91f-43b7-4657-92ba-35910b9ce300" /> |



---

### 🔸 Dataset extern

| Model | Acuratețe | CNV | DRUSEN | NORMAL |
|------|----------|-----|--------|--------|
| V1 | ~66.11% | 64% | 55% | 76% |
| V2 | ~65.97% | 69% | 35% | 87% |
| OIMHS | ~62.46% | 88% | 57% | 45% |





---



## 🏁 Concluzie generală

- Segmentarea explicită îmbunătățește interpretabilitatea modelelor
- ResNet18 oferă cea mai bună performanță și generalizare
- Pipeline-ul complet (segmentare + clasificare) simulează un flux clinic automat

---

## 💻 Interfață Streamlit pentru analiza OCT

Am creat o interfață interactivă cu **Streamlit** care permite rularea rapidă a fluxurilor de segmentare și clasificare pe imaginile OCT, pentru a verifica ușor performanța modelelor și a vizualiza rezultatele.  

### Funcționalități principale

- **Selectarea fluxului de lucru:**  
  - Tip 1 (V1) – segmentare manuală / preprocesare + clasificare  
  - Tip 2 (V2) – segmentare manuală îmbunătățită + clasificare  
  - Segmentare cu model direct – U-Net pentru mască + ResNet18 pentru clasificare  
  - Model direct (.pth) – ResNet18 pe imagine brută  
  - Model direct (.h5) – MobileNetV2 pe imagine brută cu preprocesare custom  

- **Încărcare imagine:** utilizatorul poate încărca fișiere `.jpg`, `.png` sau `.jpeg`.

- **Vizualizare rezultat:**  
  - Imagine originală  
  - Masca segmentată sau overlay-ul rezultatului segmentării  
  - Diagnostic AI (clasă prezisă + scor de încredere)  
  - Grafic cu probabilitățile fiecărei clase (pentru MobileNetV2)  

- **Procesare și inferență:**  
  - Fluxurile Tip 1 și Tip 2 folosesc funcții custom pentru segmentare și clasificare cu ResNet18  
  - Fluxul “Segmentare cu model direct” rulează mai întâi U-Net pentru mască, apoi ResNet18 pentru clasificarea pe măști  
  - Fluxurile “Model direct” clasifică imaginea brută cu ResNet18 (.pth) sau MobileNetV2 (.h5)  

- **Rulare locală:**  
  bash: python -m streamlit run app.py




## Articole

| Nr. | Autor(i) / An | Titlul articolului / proiectului | Aplicație / Domeniu | Tehnologii utilizate | Metodologie / Abordare | Rezultate | Limitări | Comentarii suplimentare |
|-----|---------------|---------------------------------|--------------------|--------------------|----------------------|-----------|----------|------------------------|
| 1   | Saman Sotoudeh-Paima, Ata Jodeiri, Fedra Hajizadeh, Hamid Soltanian-Zadeh, 2022  |Multi-Scale Convolutional Neural Network for Automated AMD Classification using Retinal OCT Images|	Image and Video Processing, AMD Classification| Preprocesare; clasificare CNN; dataset NEH și UCS; rețea piramidală de caracteristici FPN; heatmaps |Identificare AMD uscat și umed, adică DRUSEN și neovascularizație coroidiană (CNV)|93.4%+-1.4%|Model performant, dar necesită validare pe seturi mai diverse|Cel mai apropiat de proiectul meu; https://arxiv.org/abs/2110.03002|
| 2   | Aaron Chang, Alex Lo, Anusha Saha, Vivan Shah, Viswajith Rajagopalan, Aditya Rathod, 2020  |Convolutional Neural Network to Classify Retinal OCT Images|Classify Retinal OCT Images, CNN| Clasificare CNN; Keras; Dataset din instituții medicale;  Comparare cu alte modele; |Clasificare imagini în CNV, DME, DRUSEN, NORMAL| 93,75%; Testat de mine pe 91.140 imagini- 90,18%| Lipsesc unele detalii din construirea modelului| Am putut testa modelul pentru ca este public; https://github.com/ACM-Research/image-classification-cnn/tree/master?tab=readme-ov-file|
| 3   | Botond Fazekas, Guilherme Aresta, Philipp Seeböck, Julia Mai, Ursula Schmidt-Erfurth, Hrvoje Bogunović 2025| SD-RetinaNet: Topologically Constrained Semi-Supervised Retinal Lesion and Layer Segmentation in OCT| 	Computer Vision and Pattern Recognition| CNN, învăţare semi‑supravegheată, segmentare, constrângeri topologice | Segmentarea automată a leziunilor retiniene și a straturilor retinei din imaginile OCT | Înbunătățirea rezultatelor| Nu sunt publice niște rezultate precise | Ajută la segmentare; https://arxiv.org/abs/2509.20864| 
| 4   | Robbie Holland, Thomas R. P. Taylor, Christopher Holmes, Sophie Riedl, Julia Mai,..., 2025| Specialized curricula for training vision-language models in retinal image analysis| 	Artificial Intelligence, Retinal image analysis| Encoder vizual: bazat pe ResNet‑50, pre‑antrenat prin învăţare contrastivă (BYOL) pe imagini OCT, Llama 3 de la Meta, Vision-language models| Un model vizual‑lingvistic specializat (VLM) numit RetinaVLM, capabil să interpreteze imagini OCT retiniene şi să genereze rapoarte textuale pentru managementul AMD| 63-67%| Nu există cod, rezultate decente| Interesantă ideea, https://arxiv.org/abs/2209.12468|
| 5   | M Elsharkawy, A Sharafeldeen, F Khalifa, A Soliman, A Elnakib, M Ghazal, A Sewelam, A Thanos, H S Sandhu, A El-Baz, 2024| A Clinically Explainable AI-Based Grading System for Age-Related Macular Degeneration Using Optical Coherence Tomography| Biomedical and health informatics, Foundation Model| Backbone pre‑antrenat, strategie de thresholding, segmentare| Dezvoltă un model de tip foundation model cu estimare a incertitudinii (FMUE) pentru diagnosticul bolilor retiniene din imagini Optical Coherence Tomography (OCT)| 95.74%, cu strategia de thresholding, performanţa a crescut la ~97.44%| Poate necesita validare suplimentară pe populaţii diversificate şi aparate OCT diferite| Fiabilitatea modelului depinde de calitatea imaginilor, https://ieeexplore.ieee.org/document/10402009|
| 6   | Reza Rasti, Member, IEEE; Armin Biglari; Mohammad Rezapourian; Ziyun Yang; and Sina Farsiu, Fellow, IEEE, 2023| RetiFluidNet: A Self-Adaptive and Multi-Attention Deep Convolutional Network for Retinal OCT Fluid Segmentation| Transactions on medical imaging, Medical image segmentation, fluid segmentation| CNN, scheme de multi‑scale deep self‑supervision learning (DSL), conexiuni skip multiple bazate pe atenţie| Scopul este de a detecta şi segmenta acumulările de lichid / fluide, care sunt semne importante în afecţiuni| Arhitectura RetiFluidNet oferă performanțe superioare, 93.6%| Modelul a fost testat pe un singur tip de aparat OCT| Complexitate ridicată a arhitecturii https://arxiv.org/abs/2209.12468|


## 📚 Bibliografie

### Dataset-uri utilizate
- Kermany OCT 2018 Dataset – [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)  
- NEH UT OCT Dataset – [Kaggle](https://www.kaggle.com/datasets/orvile/neh-ut-oct-dataset)  
- OIMHS Dataset pentru segmentare – [Springer Nature Figshare](https://springernature.figshare.com/articles/dataset/OIMHS_dataset/23508453)

### Resurse AI și modele pre-antrenate
- **MobileNetV2** – TensorFlow / Keras, ImageNet weights  
- **ResNet18** – PyTorch, ImageNet weights  
- **U-Net** – PyTorch, arhitectură segmentare semantică
- diverse tool-uri AI

### Articole și referințe de inspirație
- Articole privind clasificarea AMD folosind CNN și segmentarea OCT  
- Documentații oficiale TensorFlow și PyTorch pentru preprocesare și antrenarea modelelor

### Alte resurse utile
- Documentații Streamlit pentru interfața interactivă  
- Coduri și exemple de procesare imagini medicale și vizualizare rezultate


