import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==== Setări generale ====
kernel_size = 3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
black_thresh = 60
intervals = [0, 80, 110, 110,160, 256]
color_map = [
    (0, 0, 0),         # negru
    (0, 255, 0),       # verde
    (255, 0, 0),       # roșu
    (0, 0, 255),       # albastru
    (250,0,250),       # roz
]

# ==== Funcție procesare imagine ====
def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imaginea nu a fost găsită: {img_path}")
    H, W = img.shape

    # === Contrast CLAHE doar pentru pixelii > black_thresh ===
    mask_nonblack = img > black_thresh
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_img = img.copy()
    contrast_img[mask_nonblack] = clahe.apply(img)[mask_nonblack]
    processed_img = contrast_img.copy()

    # === 1️⃣ Filtrare sus/jos pe baza pixelilor negri în grayscale ===
    for x in range(W):
        col = contrast_img[:, x]
        indices_nonblack = np.where(col > black_thresh)[0]
        if len(indices_nonblack) > 0:
            top = indices_nonblack.min()
            bottom = indices_nonblack.max()
            ys = np.arange(H)
            mask_filter = (ys >= top - 30) & (ys <= bottom + 50)
            processed_img[:, x] = processed_img[:, x] * mask_filter
        else:
            processed_img[:, x] = 0

    # === 2️⃣ Creează masca color pe baza intensității filtrate ===
    mask_color = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(len(intervals)-1):
        mask = (processed_img >= intervals[i]) & (processed_img < intervals[i+1])
        mask_color[mask] = color_map[i]

    # === 2.5️⃣ Completează verde între verde dacă nu e albastru între ele (fără să acopere roșu) ===
    for x in range(W):
        col = mask_color[:, x]
        green_indices = np.where(col[:,1]==255)[0]
        blue_indices  = np.where(col[:,2]==255)[0]
        red_indices   = np.where(col[:,0]==255)[0]

        if len(green_indices) > 1:
            for i in range(len(green_indices)-1):
                start = green_indices[i]
                end   = green_indices[i+1]
                if not np.any(np.isin(np.arange(start+1, end), blue_indices)):
                    for y in range(start+1, end):
                        if y not in red_indices:
                            col[y,:] = (0, 255, 0)

        # === Extindere albastru până la margini dacă nu este legat de margine (fără să acopere roșu) ===
        col_blue_indices = np.where(col[:,2]==255)[0]
        if len(col_blue_indices) > 0:
            top_blue = col_blue_indices.min()
            bottom_blue = col_blue_indices.max()
            for y in range(0, top_blue):
                if y not in red_indices:
                    col[y,:] = (80,10,10)
            for y in range(bottom_blue+1, H):
                if y not in red_indices:
                    col[y,:] = (80,10,10)

        mask_color[:, x] = col

    # === 3️⃣ Filtrare suplimentară sus/albastru și jos/rosu pe masca color ===
    blue_mask = (mask_color[:,:,1] == 255)
    red_mask  = (mask_color[:,:,1] == 255)
    new_mask = mask_color.copy()

    for x in range(W):
        col_blue = np.where(blue_mask[:, x])[0]
        if len(col_blue) > 0:
            top_blue = col_blue.min()
            ys = np.arange(H)
            mask_filter_blue = ys >= top_blue - 30
        else:
            mask_filter_blue = np.ones(H, dtype=bool)

        col_red = np.where(red_mask[:, x])[0]
        if len(col_red) > 0:
            bottom_red = col_red.max()
            ys = np.arange(H)
            mask_filter_red = ys <= bottom_red + 40
        else:
            mask_filter_red = np.ones(H, dtype=bool)

        final_mask = mask_filter_blue & mask_filter_red
        new_mask[:, x] = new_mask[:, x] * final_mask[:, None]

    # === 4️⃣ Transformăm pixelii rămași negri în verde (fără să atingem roșu sau albastru) ===
    black_pixels = (new_mask[:,:,0]==0) & (new_mask[:,:,1]==0) & (new_mask[:,:,2]==0)
    new_mask[black_pixels] = (0, 225, 0)

    green_pixels = (new_mask[:,:,0] == 0) & (new_mask[:,:,1] >= 200) & (new_mask[:,:,2] == 0)
    new_mask[green_pixels] = (0,0,0)

    brown_pixels = (new_mask[:,:,0] == 80) & (new_mask[:,:,1] == 10) & (new_mask[:,:,2] == 10)
    new_mask[brown_pixels] = (0,0,0)

    # ==== Roz ====
    pink_mask = (new_mask[:,:,0] > 200) & (new_mask[:,:,1] < 80) & (new_mask[:,:,2] > 200)
    pink_uint8 = pink_mask.astype(np.uint8) * 255
    pink_dilated = cv2.dilate(pink_uint8, kernel)
    new_pink_zone = (pink_dilated == 255) & (~pink_mask)

    blue_mask = (new_mask[:,:,0] < 50) & (new_mask[:,:,1] < 50) & (new_mask[:,:,2] > 200)
    new_pink_zone[blue_mask] = False
    new_mask[new_pink_zone] = (255, 0, 255)

    # ==== Albastru ====
    blue_uint8 = blue_mask.astype(np.uint8) * 255
    blue_dilated = cv2.dilate(blue_uint8, kernel)
    new_blue_zone = (blue_dilated == 255) & (~blue_mask)

    pink_mask = (new_mask[:,:,0] > 200) & (new_mask[:,:,1] < 80) & (new_mask[:,:,2] > 200)
    new_blue_zone[pink_mask] = False
    new_mask[new_blue_zone] = (0, 0, 255)

    # ==== Netezire ====
    smoothed = cv2.medianBlur(new_mask, 5)
    smoothed = cv2.GaussianBlur(smoothed, (3,5), 0)
    new_mask = smoothed

    # ==== Overlay ====
    orig_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(orig_rgb, 0.6, new_mask, 0.4, 0)

    return processed_img, new_mask, overlay

