# ==============================================================================
# SKRIP UNTUK PREDIKSI, EVALUASI & VISUALISASI
# File: predict_evaluate_visualize.py
# Deskripsi:
# Skrip ini memuat model U-Net 3D yang sudah dilatih, menjalankannya pada
# set data validasi untuk evaluasi kuantitatif, dan memvisualisasikan
# hasil prediksi serta distribusi skor performa.
# ==============================================================================

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

# ==============================================================================
# BAGIAN 1: DEFINISI FUNGSI CUSTOM & DATA GENERATOR
# Catatan: Bagian ini harus identik dengan yang ada di skrip training.
# ==============================================================================
SMOOTH = 1e-6

# --- Nama Kelas & Konfigurasi ---
CLASS_NAMES = {
    0: "Background",
    1: "Necrotic and non-enhancing tumor core (NCR/NET)",
    2: "Peritumoral edema (ED)",
    3: "enhancing tumor (ET)"
}
NUM_CLASSES = 4

# --- Fungsi Metrik dan Loss (Versi TensorFlow/Keras) ---
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def iou_score_func(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + SMOOTH) / (union + SMOOTH)

def mean_iou(y_true, y_pred):
    total_iou = 0.0
    for i in range(NUM_CLASSES):
        total_iou += iou_score_func(y_true[..., i], y_pred[..., i])
    return total_iou / NUM_CLASSES

def create_dice_coef_per_class(class_id, class_name):
    def dice_coef_class(y_true, y_pred):
        return dice_coef(y_true[..., class_id], y_pred[..., class_id])
    dice_coef_class.__name__ = f'dice_{class_name}'
    return dice_coef_class

def create_iou_score_per_class(class_id, class_name):
    def iou_score_class(y_true, y_pred):
        return iou_score_func(y_true[..., class_id], y_pred[..., class_id])
    iou_score_class.__name__ = f'iou_{class_name}'
    return iou_score_class

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred)
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
    return K.sum(loss, axis=-1)

def create_weighted_dice_loss(class_weights):
    def weighted_dice_loss(y_true, y_pred):
        total_loss = 0
        for i in range(len(class_weights)):
            loss_per_class = 1 - dice_coef(y_true[..., i], y_pred[..., i])
            total_loss += class_weights[i] * loss_per_class
        return total_loss
    return weighted_dice_loss

def create_combo_loss(class_weights, dice_weight=0.5, focal_weight=0.5):
    _weighted_dice = create_weighted_dice_loss(class_weights)
    def combo_loss(y_true, y_pred):
        dice_l = _weighted_dice(y_true, y_pred)
        focal_l = focal_loss(y_true, y_pred, gamma=2.0, alpha=class_weights)
        return (dice_weight * dice_l) + (focal_weight * focal_l)
    return combo_loss

# --- Fungsi Metrik (Versi Numpy untuk Evaluasi) ---
def numpy_dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    sum_of_masks = np.sum(y_true) + np.sum(y_pred)
    if sum_of_masks == 0: return 1.0
    return (2. * intersection + SMOOTH) / (sum_of_masks + SMOOTH)

def numpy_iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    if union == 0: return 1.0
    return (intersection + SMOOTH) / (union + SMOOTH)

# --- Data Generator (Tidak Berubah) ---
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_pairs, batch_size=1, dim=(128,128,128), n_channels=3,
                 n_classes=4, shuffle=True):
        self.dim, self.batch_size = dim, batch_size
        self.data_pairs, self.n_channels = data_pairs, n_channels
        self.n_classes, self.shuffle = n_classes, shuffle
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.data_pairs) / self.batch_size))
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_pairs = [self.data_pairs[k] for k in indexes]
        X, y = self.__data_generation(batch_pairs)
        return X, y
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    def __data_generation(self, batch_pairs):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))
        for i, (img_path, mask_path) in enumerate(batch_pairs):
            X[i,] = np.load(img_path)
            y[i,] = np.load(mask_path)
        return X, y

# ==============================================================================
# BAGIAN 2: KONFIGURASI DAN PERSIAPAN DATA
# ==============================================================================
# --- Konfigurasi ---
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 128, 128, 128
IMG_CHANNELS, BATCH_SIZE = 3, 1

# --- Siapkan path data dengan metode split 70/10/20 ---
BRATS_2020_DIR = "C:/Aulia/BraTS 2020/Data 2020/"
BRATS_2021_DIR = "C:/Aulia/BraTS 2021/FullData/"
all_image_paths = []
all_image_paths.extend(glob.glob(os.path.join(BRATS_2020_DIR, 'train/images', '*.npy')))
all_image_paths.extend(glob.glob(os.path.join(BRATS_2020_DIR, 'val/images', '*.npy')))
all_image_paths.extend(glob.glob(os.path.join(BRATS_2021_DIR, 'train/images', '*.npy')))
all_image_paths.extend(glob.glob(os.path.join(BRATS_2021_DIR, 'val/images', '*.npy')))
all_mask_paths = [p.replace('images', 'masks').replace('image_', 'mask_') for p in all_image_paths]
all_data_pairs = list(zip(all_image_paths, all_mask_paths))

# --- Langkah 1: Pisahkan data training (80%) dan testing (20%) ---
# Kita ambil 80% dulu untuk nanti dibagi lagi menjadi training dan validasi.
train_val_pairs, test_pairs = train_test_split(all_data_pairs, test_size=0.20, random_state=42)

# --- Langkah 2: Pisahkan 80% data tadi menjadi training (70%) dan validasi (10%) ---
# Proporsi validasi dari sisa 80% adalah 10/80 = 0.125
train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.125, random_state=42) 

print(f"Total data ditemukan: {len(all_data_pairs)}")
print(f"Jumlah data training (70%): {len(train_pairs)}")
print(f"Jumlah data validasi (10%): {len(val_pairs)}")
print(f"Jumlah data testing (20%): {len(test_pairs)}")


# --- Buat validation generator (shuffle=False agar urutan data konsisten) ---
# Skrip ini fokus pada evaluasi di set VALIDASI
val_generator = DataGenerator(val_pairs, BATCH_SIZE, (IMG_DEPTH, IMG_WIDTH, IMG_HEIGHT), IMG_CHANNELS, NUM_CLASSES, shuffle=False)
print(f"Data validasi siap untuk dievaluasi: {len(val_generator)} sampel.")

# ==============================================================================
# BAGIAN 3: MUAT MODEL YANG SUDAH DILATIH
# ==============================================================================
MODEL_PATH = 'C:/Aulia/Kumpulan Eksperimen/Eksperimen 17/Model/Unet_best.h5'
my_model = None

if os.path.exists(MODEL_PATH):
    print(f"\nModel ditemukan di {MODEL_PATH}. Memuat model...")
    # --- Siapkan Custom Objects yang LENGKAP ---
    class_weights = np.array([0.004483, 0.391893, 0.152227, 0.45139])
    combo_loss_fn = create_combo_loss(class_weights=class_weights, dice_weight=0.6, focal_weight=0.4)
    
    custom_objects = {
        'combo_loss': combo_loss_fn,
        'dice_coef': dice_coef,
        'mean_iou': mean_iou,
    }
    # Gunakan nama kelas asli tanpa modifikasi agar cocok dengan model yang disimpan
    for i in range(NUM_CLASSES):
        # Ambil nama kelas apa adanya dari dictionary
        class_name = CLASS_NAMES[i]
        
        if i > 0: # Dice hanya untuk kelas tumor
            # Buat nama metrik persis seperti di skrip training
            dice_metric_name = f'dice_{class_name}'
            custom_objects[dice_metric_name] = create_dice_coef_per_class(i, class_name)
        
        iou_metric_name = f'iou_{class_name}'
        custom_objects[iou_metric_name] = create_iou_score_per_class(i, class_name)
    # =======================================================================
        
    my_model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model berhasil dimuat.")
else:
    print(f"❌ Model tidak ditemukan di path: {MODEL_PATH}. Skrip akan berhenti.")

# ==============================================================================
# BAGIAN 4: EVALUASI KUANTITATIF MODEL
# ==============================================================================

if my_model and len(val_generator) > 0:
    print("\n--- Memulai Evaluasi Kuantitatif pada Seluruh Data Validasi ---")
    
    # Inisialisasi list untuk menyimpan skor
    all_class_dice_scores = [[] for _ in range(NUM_CLASSES)]
    all_class_iou_scores = [[] for _ in range(NUM_CLASSES)]
    wt_dices, tc_dices, et_dices = [], [], []

    # Loop melalui semua data di validation generator
    for i in tqdm(range(len(val_generator)), desc="Mengevaluasi"):
        img_batch, mask_batch = val_generator[i]
        pred_batch = my_model.predict(img_batch, verbose=0)
        
        true_mask = np.argmax(mask_batch[0], axis=-1)
        pred_mask = np.argmax(pred_batch[0], axis=-1)

        # 1. Hitung metrik untuk setiap KELAS
        for cls in range(NUM_CLASSES):
            y_true_cls = (true_mask == cls)
            y_pred_cls = (pred_mask == cls)
            all_class_dice_scores[cls].append(numpy_dice_score(y_true_cls, y_pred_cls))
            all_class_iou_scores[cls].append(numpy_iou_score(y_true_cls, y_pred_cls))
            
        # 2. Hitung metrik untuk setiap REGION (Standar BraTS)
        wt_true = np.isin(true_mask, [1, 2, 3]); wt_pred = np.isin(pred_mask, [1, 2, 3])
        wt_dices.append(numpy_dice_score(wt_true, wt_pred))

        tc_true = np.isin(true_mask, [1, 3]); tc_pred = np.isin(pred_mask, [1, 3])
        tc_dices.append(numpy_dice_score(tc_true, tc_pred))

        et_true = (true_mask == 3); et_pred = (pred_mask == 3)
        et_dices.append(numpy_dice_score(et_true, et_pred))

    # --- Tampilkan Hasil Rata-rata ---
    print("\n\n===========================================================")
    print("--- HASIL EVALUASI RATA-RATA PADA SELURUH DATA VALIDASI ---")
    print("===========================================================")
    
    print("\nEvaluasi Rata-rata per KELAS:")
    print(f"{'Kelas':<40} | {'Mean Dice':<15} | {'Mean IoU':<15}")
    print("-" * 75)
    for cls in range(NUM_CLASSES):
        mean_dice = np.mean(all_class_dice_scores[cls])
        mean_iou = np.mean(all_class_iou_scores[cls])
        label = f"Kelas {cls} ({CLASS_NAMES[cls]})"
        print(f"{label:<40} | {mean_dice:<15.4f} | {mean_iou:<15.4f}")
        
    print("\nEvaluasi Rata-rata per REGION (Standar BraTS):")
    print(f"{'Region':<25} | {'Mean Dice':<15}")
    print("-" * 45)
    print(f"{'Whole Tumor (WT)':<25} | {np.mean(wt_dices):<15.4f}")
    print(f"{'Tumor Core (TC)':<25} | {np.mean(tc_dices):<15.4f}")
    print(f"{'Enhancing Tumor (ET)':<25} | {np.mean(et_dices):<15.4f}")
    print("===========================================================\n")

# ==============================================================================
# BAGIAN 5: PLOT GRAFIK DISTRIBUSI SKOR
# ==============================================================================
    print("\n--- Membuat Grafik Distribusi Skor ---")

    # --- Grafik 1: Box Plot Distribusi Skor per KELAS ---
    plt.figure(figsize=(18, 8))
    plt.suptitle("Distribusi Skor Evaluasi per Kelas", fontsize=18)
    class_labels = [f"{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)]

    # Box plot untuk Dice Score per KELAS
    plt.subplot(1, 2, 1)
    plt.boxplot(all_class_dice_scores, labels=class_labels, patch_artist=True)
    plt.title('Distribusi Dice Score per Kelas', fontsize=14)
    plt.ylabel('Dice Score', fontsize=12)
    plt.xticks(rotation=10, ha="right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.02)

    # Box plot untuk IoU Score per KELAS
    plt.subplot(1, 2, 2)
    plt.boxplot(all_class_iou_scores, labels=class_labels, patch_artist=True)
    plt.title('Distribusi IoU Score per Kelas', fontsize=14)
    plt.ylabel('IoU Score (Jaccard Index)', fontsize=12)
    plt.xticks(rotation=10, ha="right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.02)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # --- Grafik 2: Box Plot Distribusi Skor per REGION ---
    plt.figure(figsize=(10, 7))
    plt.suptitle("Distribusi Skor Evaluasi per Region Tumor (Standar BraTS)", fontsize=16)
    region_labels = ['Whole Tumor (WT)', 'Tumor Core (TC)', 'Enhancing Tumor (ET)']
    
    plt.boxplot([wt_dices, tc_dices, et_dices], labels=region_labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
    plt.title('Distribusi Dice Score per Region', fontsize=14)
    plt.ylabel('Dice Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.02)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ==============================================================================
# BAGIAN 6: VISUALISASI PREDIKSI PADA SATU CONTOH
# ==============================================================================
if my_model and len(val_generator) > 0:
    print("\n--- Memulai Visualisasi Contoh Prediksi ---")
    
    # Ambil satu gambar secara acak untuk divisualisasikan
    sample_index = 88
    print(f"Menampilkan hasil untuk sampel validasi indeks ke-{sample_index}...")
    
    # Ambil gambar, prediksi, dan proses hasilnya
    test_img, test_mask = val_generator[sample_index]
    test_pred = my_model.predict(test_img)
    test_img_single = test_img[0]
    test_mask_argmax = np.argmax(test_mask[0], axis=-1)
    test_pred_argmax = np.argmax(test_pred[0], axis=-1)

    # Cari irisan terbaik (slice dengan area tumor terbesar di ground truth)
    tumor_mask = test_mask_argmax > 0
    slice_with_most_tumor = np.argmax(np.sum(tumor_mask, axis=(0, 1))) if np.sum(tumor_mask) > 0 else test_img_single.shape[2] // 2
    n_slice = slice_with_most_tumor
    print(f"Menampilkan irisan (slice) terbaik: {n_slice}")

    # --- Tampilkan hasil ---
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Hasil Prediksi pada Sampel #{sample_index} (Slice {n_slice})', fontsize=16)
    
    plt.subplot(231); plt.imshow(test_img_single[:, :, n_slice, 0], cmap='gray'); plt.title('Input: Flair'); plt.axis('off')
    plt.subplot(232); plt.imshow(test_img_single[:, :, n_slice, 1], cmap='gray'); plt.title('Input: T1ce'); plt.axis('off')
    plt.subplot(233); plt.imshow(test_img_single[:, :, n_slice, 2], cmap='gray'); plt.title('Input: T2'); plt.axis('off')
    
    # 'viridis' adalah colormap yang baik untuk label kategori
    plt.subplot(234); plt.imshow(test_mask_argmax[:, :, n_slice], cmap='viridis', vmin=0, vmax=NUM_CLASSES-1); plt.title('Ground Truth Mask'); plt.axis('off')
    plt.subplot(235); plt.imshow(test_pred_argmax[:, :, n_slice], cmap='viridis', vmin=0, vmax=NUM_CLASSES-1); plt.title('Predicted Mask'); plt.axis('off')
    
    # Overlay prediksi pada gambar flair
    plt.subplot(236)
    plt.imshow(test_img_single[:, :, n_slice, 0], cmap='gray')
    plt.imshow(np.ma.masked_where(test_pred_argmax[:, :, n_slice] == 0, test_pred_argmax[:, :, n_slice]), cmap='viridis', alpha=0.6)
    plt.title('Overlay: Prediksi pada Flair'); plt.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
# ==============================================================================
# BAGIAN 7: MEMBUAT VIDEO VISUALISASI PREDIKSI
# ==============================================================================
import cv2
OUTPUT_DIR = 'C:/Aulia/BraTS 2020 dan 2021/Unet attention/Model/training-attn-unet/'
if my_model and len(val_generator) > 0:
    print("\n--- Memulai Pembuatan Video Visualisasi Prediksi ---")
    
    # Menggunakan data yang sama dengan yang ditampilkan di BAGIAN 6
    # 'sample_index', 'test_img_single', 'test_mask_argmax', 'test_pred_argmax'
    # sudah tersedia dari blok kode sebelumnya.
    
    video_frames = [] # List untuk menyimpan frame video di memori

    # Loop melalui setiap slice dari volume 3D
    for n_slice in tqdm(range(IMG_DEPTH), desc="Membuat Frame Video"):
        # Buat figure untuk setiap frame
        fig = plt.figure(figsize=(15, 10))
        plt.suptitle(f'Hasil Prediksi pada Sampel #{sample_index} (Slice {n_slice})', fontsize=16)
        
        # Plot 1: Flair
        plt.subplot(231); plt.imshow(test_img_single[:, :, n_slice, 0], cmap='gray'); plt.title('Input: Flair'); plt.axis('off')
        
        # Plot 2: T1ce
        plt.subplot(232); plt.imshow(test_img_single[:, :, n_slice, 1], cmap='gray'); plt.title('Input: T1ce'); plt.axis('off')

        # Plot 3: T2
        plt.subplot(233); plt.imshow(test_img_single[:, :, n_slice, 2], cmap='gray'); plt.title('Input: T2'); plt.axis('off')
        
        # Plot 4: Ground Truth Mask
        plt.subplot(234); plt.imshow(test_mask_argmax[:, :, n_slice], cmap='viridis', vmin=0, vmax=NUM_CLASSES-1); plt.title('Ground Truth Mask'); plt.axis('off')

        # Plot 5: Predicted Mask
        plt.subplot(235); plt.imshow(test_pred_argmax[:, :, n_slice], cmap='viridis', vmin=0, vmax=NUM_CLASSES-1); plt.title('Predicted Mask'); plt.axis('off')
        
        # Plot 6: Overlay
        plt.subplot(236)
        plt.imshow(test_img_single[:, :, n_slice, 0], cmap='gray')
        plt.imshow(np.ma.masked_where(test_pred_argmax[:, :, n_slice] == 0, test_pred_argmax[:, :, n_slice]), cmap='viridis', alpha=0.6)
        plt.title('Overlay: Prediksi pada Flair'); plt.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Konversi plot matplotlib ke array numpy untuk video
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # Konversi RGB ke BGR untuk OpenCV
        
        # Tutup plot untuk menghemat memori
        plt.close(fig)

    # --- Gabungkan semua frame menjadi video ---
    if video_frames:
        video_filename = f"prediksi_sampel_{sample_index}.mp4"
        video_path = os.path.join(OUTPUT_DIR, video_filename)
        
        # Ambil dimensi dari frame pertama
        height, width, _ = video_frames[0].shape
        
        # Inisialisasi VideoWriter
        # 'mp4v' adalah codec yang umum untuk file .mp4
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height)) # 10 FPS
        
        for frame in tqdm(video_frames, desc="Menyimpan Video"):
            video_writer.write(frame)
        
        video_writer.release()
        print(f"✅ Video visualisasi berhasil disimpan di: {video_path}")
    else:
        print("❌ Tidak ada frame yang dibuat, video tidak dapat disimpan.")
