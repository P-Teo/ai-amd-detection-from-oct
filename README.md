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
   - 
### 🧩 Faza 2 – Clasificare cu segmentare explicită

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

---

### 🔸 Dataset extern

| Model | Acuratețe | CNV | DRUSEN | NORMAL |
|------|----------|-----|--------|--------|
| MobileNetV2 | ~58.46% | 76% | 11% | 96% |
| ResNet18 | ~74.06% | 93% | 45% | 96% |
| Model GitHub | ~43% | - | - | - |

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

## 🏁 Concluzie generală

- Segmentarea explicită îmbunătățește interpretabilitatea modelelor
- ResNet18 oferă cea mai bună performanță și generalizare
- Pipeline-ul complet (segmentare + clasificare) simulează un flux clinic automat







## Articole

| Nr. | Autor(i) / An | Titlul articolului / proiectului | Aplicație / Domeniu | Tehnologii utilizate | Metodologie / Abordare | Rezultate | Limitări | Comentarii suplimentare |
|-----|---------------|---------------------------------|--------------------|--------------------|----------------------|-----------|----------|------------------------|
| 1   | Saman Sotoudeh-Paima, Ata Jodeiri, Fedra Hajizadeh, Hamid Soltanian-Zadeh, 2022  |Multi-Scale Convolutional Neural Network for Automated AMD Classification using Retinal OCT Images|	Image and Video Processing, AMD Classification| Preprocesare; clasificare CNN; dataset NEH și UCS; rețea piramidală de caracteristici FPN; heatmaps |Identificare AMD uscat și umed, adică DRUSEN și neovascularizație coroidiană (CNV)|93.4%+-1.4%|Model performant, dar necesită validare pe seturi mai diverse|Cel mai apropiat de proiectul meu; https://arxiv.org/abs/2110.03002|
| 2   | Aaron Chang, Alex Lo, Anusha Saha, Vivan Shah, Viswajith Rajagopalan, Aditya Rathod, 2020  |Convolutional Neural Network to Classify Retinal OCT Images|Classify Retinal OCT Images, CNN| Clasificare CNN; Keras; Dataset din instituții medicale;  Comparare cu alte modele; |Clasificare imagini în CNV, DME, DRUSEN, NORMAL| 93,75%; Testat de mine pe 91.140 imagini- 90,18%| Lipsesc unele detalii din construirea modelului| Am putut testa modelul pentru ca este public; https://github.com/ACM-Research/image-classification-cnn/tree/master?tab=readme-ov-file|
| 3   | Botond Fazekas, Guilherme Aresta, Philipp Seeböck, Julia Mai, Ursula Schmidt-Erfurth, Hrvoje Bogunović 2025| SD-RetinaNet: Topologically Constrained Semi-Supervised Retinal Lesion and Layer Segmentation in OCT| 	Computer Vision and Pattern Recognition| CNN, învăţare semi‑supravegheată, segmentare, constrângeri topologice | Segmentarea automată a leziunilor retiniene și a straturilor retinei din imaginile OCT | Înbunătățirea rezultatelor| Nu sunt publice niște rezultate precise | Ajută la segmentare; https://arxiv.org/abs/2509.20864| 
| 4   | Robbie Holland, Thomas R. P. Taylor, Christopher Holmes, Sophie Riedl, Julia Mai,..., 2025| Specialized curricula for training vision-language models in retinal image analysis| 	Artificial Intelligence, Retinal image analysis| Encoder vizual: bazat pe ResNet‑50, pre‑antrenat prin învăţare contrastivă (BYOL) pe imagini OCT, Llama 3 de la Meta, Vision-language models| Un model vizual‑lingvistic specializat (VLM) numit RetinaVLM, capabil să interpreteze imagini OCT retiniene şi să genereze rapoarte textuale pentru managementul AMD| 63-67%| Nu există cod, rezultate decente| Interesantă ideea, https://arxiv.org/abs/2209.12468|
| 5   | M Elsharkawy, A Sharafeldeen, F Khalifa, A Soliman, A Elnakib, M Ghazal, A Sewelam, A Thanos, H S Sandhu, A El-Baz, 2024| A Clinically Explainable AI-Based Grading System for Age-Related Macular Degeneration Using Optical Coherence Tomography| Biomedical and health informatics, Foundation Model| Backbone pre‑antrenat, strategie de thresholding, segmentare| Dezvoltă un model de tip foundation model cu estimare a incertitudinii (FMUE) pentru diagnosticul bolilor retiniene din imagini Optical Coherence Tomography (OCT)| 95.74%, cu strategia de thresholding, performanţa a crescut la ~97.44%| Poate necesita validare suplimentară pe populaţii diversificate şi aparate OCT diferite| Fiabilitatea modelului depinde de calitatea imaginilor, https://ieeexplore.ieee.org/document/10402009|
| 6   | Reza Rasti, Member, IEEE; Armin Biglari; Mohammad Rezapourian; Ziyun Yang; and Sina Farsiu, Fellow, IEEE, 2023| RetiFluidNet: A Self-Adaptive and Multi-Attention Deep Convolutional Network for Retinal OCT Fluid Segmentation| Transactions on medical imaging, Medical image segmentation, fluid segmentation| CNN, scheme de multi‑scale deep self‑supervision learning (DSL), conexiuni skip multiple bazate pe atenţie| Scopul este de a detecta şi segmenta acumulările de lichid / fluide, care sunt semne importante în afecţiuni| Arhitectura RetiFluidNet oferă performanțe superioare, 93.6%| Modelul a fost testat pe un singur tip de aparat OCT| Complexitate ridicată a arhitecturii https://arxiv.org/abs/2209.12468|


## Schema și etapele proiectului

<table>
  <tr>
    <td>
      <img src="images/PI.png" alt="Workflow proiect OCT" width="400">
    </td>
    <td>
      <p></p> <br> 
      <p><span style="color:#ff7f00;"><strong>1️. Colectarea datelor:</strong></span> Alegem un dataset popular cu cât mai multe imagini (ex. Kermany2018 și Srinivasan2014). Folosim un dataset cu imagini OCT în 3 foldere: train, val, test.​</p> <br> 
      <p><span style="color:#ff7f00;"><strong>2️. Preprocesarea imaginilor:</strong></span> Redimensionare, conversie RGB/grayscale, normalizare pixeli (0–1), eliminare zgomot și augmentare (rotații, flip, shift).</p> <br> 
      <p><span style="color:#ff7f00;"><strong>3️. Definire Model CNN (MobileNetV2):</strong></span> Definim modelul MobileNetV2 ca bază și adăugăm un strat final de clasificarea a celor 3 clase. Modelul este folosit ca extractor de caracteristici.​</p> <br> 
      <p><span style="color:#ff7f00;"><strong>4️. Compilare și antrenare model:</strong></span> Antrenăm modelul pe datele din train pentru 10 epoci. La sfârșitul fiecărei epoci se evaluează val și se raportează loss/accuracy pentru antrenare și validare.​</p> <br> 
      <p><span style="color:#ff7f00;"><strong>5️. Testare:</strong></span> Testăm modelul pe setul de date din test și pe încă un set de date + calcul acuratețe pe fiecare clasă.</p> <br> 
      <p><span style="color:#ff7f00;"><strong>6️. Analiză rezultate:</strong> </span>Grafice Accuracy/Loss și obținem precizia, sensibilitatea, scorul F1 și matricea de confuzie​</p> <br> 
    </td>
  </tr>
</table>




