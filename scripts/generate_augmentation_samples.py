import cv2
import numpy as np
import os

def generate_samples():
    base_dir = "/media/dinhdat/OD/DOANTOTNGHIEP/DOANTOTNGHIEP/archive/preprocessed_images"
    img1_path = os.path.join(base_dir, "0_left.jpg")
    img2_path = os.path.join(base_dir, "0_right.jpg")

    # Đọc ảnh (BGR)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Cannot find the images in the path!")
        return

    # Resize về 224x224
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))

    # --- MIXUP ---
    lam_mixup = 0.5
    img_mixup = cv2.addWeighted(img1, lam_mixup, img2, 1 - lam_mixup, 0)

    # --- CUTMIX ---
    img_cutmix = img1.copy()
    lam_cutmix = 0.5
    W, H = 224, 224
    rw = int(W * np.sqrt(1 - lam_cutmix))
    rh = int(H * np.sqrt(1 - lam_cutmix))
    
    # Tâm cắt (giữa ảnh cho dễ thấy)
    cx, cy = W // 2, H // 2
    x1 = np.clip(cx - rw // 2, 0, W)
    y1 = np.clip(cy - rh // 2, 0, H)
    x2 = np.clip(cx + rw // 2, 0, W)
    y2 = np.clip(cy + rh // 2, 0, H)
    
    img_cutmix[y1:y2, x1:x2] = img2[y1:y2, x1:x2]

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img1, 'Image 1', (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img2, 'Image 2', (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img_mixup, 'MixUp (Ghosting)', (10, 30), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img_cutmix, 'CutMix (Sharp)', (10, 30), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Ghép 4 ảnh nằm ngang
    final_image = np.hstack((img1, img2, img_mixup, img_cutmix))

    # Lưu ảnh
    out_path = "/media/dinhdat/OD/DOANTOTNGHIEP/DOANTOTNGHIEP/docs/mixup_cutmix_illustration.jpg"
    cv2.imwrite(out_path, final_image)
    print(f"Saved illustration to {out_path}")

if __name__ == "__main__":
    generate_samples()
