import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, Dropout, Conv3DTranspose, concatenate,
    BatchNormalization, Activation, Add, Multiply, UpSampling3D,
    LayerNormalization, MultiHeadAttention, Dense, Reshape,
    Embedding
)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
tf.debugging.set_log_device_placement(False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

#==============================================================================
# BAGIAN 1: LOSS FUNCTIONS DAN METRICS
#==============================================================================
SMOOTH = 1e-6

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def iou_score_func(y_true, y_pred): # Mengubah nama agar tidak konflik dengan variabel
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + SMOOTH) / (union + SMOOTH)

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

def create_combo_loss(class_weights, focal_gamma=2.0, dice_weight=0.5, focal_weight=0.5):
    _weighted_dice = create_weighted_dice_loss(class_weights)
    def combo_loss(y_true, y_pred):
        dice_l = _weighted_dice(y_true, y_pred)
        focal_l = focal_loss(y_true, y_pred, gamma=focal_gamma, alpha=class_weights)
        return (dice_weight * dice_l) + (focal_weight * focal_l)
    return combo_loss

def create_dice_coef_per_class(class_id, class_name):
    def dice_coef_class(y_true, y_pred):
        y_true_class = y_true[..., class_id]
        y_pred_class = y_pred[..., class_id]
        return dice_coef(y_true_class, y_pred_class)
    dice_coef_class.__name__ = f'dice_{class_name}'
    return dice_coef_class

### PERMINTAAN GRAFIK IOU (Langkah 1): Membuat fungsi metrik IoU Score per kelas
def create_iou_score_per_class(class_id, class_name):
    def iou_score_class(y_true, y_pred):
        y_true_class = y_true[..., class_id]
        y_pred_class = y_pred[..., class_id]
        return iou_score_func(y_true_class, y_pred_class)
    iou_score_class.__name__ = f'iou_{class_name}'
    return iou_score_class

### PERMINTAAN GRAFIK IOU (Langkah 1): Membuat fungsi metrik Mean IoU
def mean_iou(y_true, y_pred):
    # Menghitung IoU untuk setiap kelas dan merata-ratakannya
    total_iou = 0.0
    num_classes = K.int_shape(y_pred)[-1]
    for i in range(num_classes):
        total_iou += iou_score_func(y_true[..., i], y_pred[..., i])
    return total_iou / num_classes

#==============================================================================
# BAGIAN 2: DATA GENERATOR (Tidak ada perubahan)
#==============================================================================
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_pairs, batch_size=1, dim=(128,128,128), n_channels=3,
                 n_classes=4, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.data_pairs = data_pairs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
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

#==============================================================================
# BAGIAN 3: DEFINISI ARSITEKTUR MODEL (Tidak ada perubahan)
#==============================================================================
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, Conv3DTranspose, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation

# Asumsi 'kernel_initializer' sudah didefinisikan di skrip Anda, contoh:
kernel_initializer = 'he_uniform'

def unet3conv_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    #Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)

    #Expansive path
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
    
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name="U-Net")
    model.summary()
    return model

#==============================================================================
# BAGIAN 4: SKRIP TRAINING UTAMA
#==============================================================================

# --- Konfigurasi dan Hyperparameters ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_DEPTH = 128
IMG_CHANNELS = 3
NUM_CLASSES = 4
INPUT_SHAPE = (IMG_DEPTH, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
LEARNING_RATE = 1e-4
EPOCHS = 100
BATCH_SIZE = 1
VALIDATION_SPLIT = 0.2

# --- Definisikan Nama Kelas untuk Pelabelan ---
CLASS_NAMES = {
    0: "Background",
    1: "Necrotic and non-enhancing tumor core (NCR/NET)",
    2: "Peritumoral edema (ED)",
    3: "enhancing tumor (ET)"
}

import os
import glob
from sklearn.model_selection import train_test_split

# --- Tentukan Path untuk Dataset BraTS 2020 dan 2021 ---
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

# --- Siapkan Data Generator ---
train_generator = DataGenerator(train_pairs, BATCH_SIZE, (IMG_DEPTH, IMG_WIDTH, IMG_HEIGHT), IMG_CHANNELS, NUM_CLASSES)
val_generator = DataGenerator(val_pairs, BATCH_SIZE, (IMG_DEPTH, IMG_WIDTH, IMG_HEIGHT), IMG_CHANNELS, NUM_CLASSES)
test_generator = DataGenerator(test_pairs, BATCH_SIZE, (IMG_DEPTH, IMG_WIDTH, IMG_HEIGHT), IMG_CHANNELS, NUM_CLASSES, shuffle=False)


OUTPUT_DIR = 'C:/Aulia/Kumpulan Eksperimen/Eksperimen 18/Model/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'AttnDropUnet_best.h5')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# --- Hitung Bobot Kelas ---
class_weights = np.array([0.004483, 0.391893, 0.152227, 0.45139])
print(f"Menggunakan bobot kelas: {class_weights}")

# --- Siapkan Data Generator ---
train_generator = DataGenerator(train_pairs, BATCH_SIZE, (IMG_DEPTH, IMG_WIDTH, IMG_HEIGHT), IMG_CHANNELS, NUM_CLASSES)
val_generator = DataGenerator(val_pairs, BATCH_SIZE, (IMG_DEPTH, IMG_WIDTH, IMG_HEIGHT), IMG_CHANNELS, NUM_CLASSES)

model =  unet3conv_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, NUM_CLASSES)

# --- Kompilasi Model ---
combo_loss_fn = create_combo_loss(class_weights=class_weights, dice_weight=0.6, focal_weight=0.4)

### GRAFIK IOU 
per_class_dice_metrics = [create_dice_coef_per_class(i, CLASS_NAMES[i]) for i in range(1, NUM_CLASSES)]
# IoU untuk semua kelas, termasuk background
per_class_iou_metrics = [create_iou_score_per_class(i, CLASS_NAMES[i]) for i in range(NUM_CLASSES)]

all_metrics = per_class_dice_metrics + per_class_iou_metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=combo_loss_fn,
              metrics=all_metrics)

# --- Siapkan Callbacks ---
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
tensorboard_callback = TensorBoard(log_dir=LOG_DIR)
callbacks_list = [checkpoint, reduce_lr, tensorboard_callback]

# --- Mulai Training ---
history = None
if len(train_generator) > 0 and len(val_generator) > 0:
    print("\nMemulai proses training... (Tekan Ctrl+C atau tombol Stop untuk menghentikan)")
    try:
        history = model.fit(train_generator,
                            steps_per_epoch=len(train_generator),
                            epochs=EPOCHS,
                            validation_data=val_generator,
                            validation_steps=len(val_generator),
                            callbacks=callbacks_list)
    except KeyboardInterrupt:
        print("\nTraining dihentikan secara manual.")
else:
    print("\nTraining dibatalkan: Tidak ada data yang ditemukan.")

#==============================================================================
# BAGIAN 5: PLOT HASIL TRAINING
#==============================================================================
if history:
    print("\n--- Membuat Grafik Hasil Training ---")
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    # Grafik 1: Loss, Overall Dice, Mean IoU
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, 'y', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_summary_plot.png'))
    plt.show()

    # Grafik 2: Dice Score per Kelas Tumor
    plt.figure(figsize=(12, 8))
    colors = ['g', 'b', 'm'] # Warna untuk kelas 1, 2, 3
    for i in range(1, NUM_CLASSES):
        class_name = CLASS_NAMES[i]
        metric_name = f'dice_{class_name}'
        val_metric_name = f'val_dice_{class_name}'
        
        plt.plot(epochs, history.history[metric_name], color=colors[i-1], linestyle='--', label=f'Training Dice ({class_name})')
        plt.plot(epochs, history.history[val_metric_name], color=colors[i-1], label=f'Validation Dice ({class_name})')

    plt.title('Training and Validation Dice Score per Tumor Class')
    plt.xlabel('Epochs'); plt.ylabel('Dice Score'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dice_per_class_plot.png'))
    plt.show()

    ### PERMINTAAN GRAFIK IOU (Langkah 3): Membuat grafik IoU Score per kelas
    plt.figure(figsize=(12, 8))
    colors = ['c', 'g', 'b', 'm'] # Warna untuk kelas 0, 1, 2, 3
    for i in range(NUM_CLASSES):
        class_name = CLASS_NAMES[i]
        metric_name = f'iou_{class_name}'
        val_metric_name = f'val_iou_{class_name}'
        
        plt.plot(epochs, history.history[metric_name], color=colors[i], linestyle='--', label=f'Training IoU ({class_name})')
        plt.plot(epochs, history.history[val_metric_name], color=colors[i], label=f'Validation IoU ({class_name})')

    plt.title('Training and Validation IoU Score per Class')
    plt.xlabel('Epochs'); plt.ylabel('IoU Score'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'iou_per_class_plot.png'))
    plt.show()

#==============================================================================
# BAGIAN 6: EVALUASI MODEL SETELAH TRAINING
#==============================================================================
print(MODEL_SAVE_PATH)

print("\n--- Memulai Evaluasi Model pada Seluruh Data Validasi ---")
if os.path.exists(MODEL_SAVE_PATH):
    ### PERMINTAAN GRAFIK IOU (Langkah 4): Memuat model dengan custom objects yang lengkap
    custom_objects = {
        'combo_loss': combo_loss_fn,
        'dice_coef': dice_coef,
        'mean_iou': mean_iou, # <-- TAMBAHAN BARU
    }
    # Tambahkan metrik per kelas (Dice dan IoU)
    for i in range(NUM_CLASSES):
        if i > 0: # Dice hanya untuk kelas tumor
            dice_metric_name = f'dice_{CLASS_NAMES[i]}'
            custom_objects[dice_metric_name] = create_dice_coef_per_class(i, CLASS_NAMES[i])
        
        iou_metric_name = f'iou_{CLASS_NAMES[i]}'
        custom_objects[iou_metric_name] = create_iou_score_per_class(i, CLASS_NAMES[i])
        
    my_model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects=custom_objects)
    
    # Inisialisasi list untuk menyimpan skor
    all_class_dice_scores = [[] for _ in range(NUM_CLASSES)]
    all_class_iou_scores = [[] for _ in range(NUM_CLASSES)]
    all_wt_scores, all_tc_scores, all_et_scores = [], [], []
    
    # Inisialisasi list untuk menyimpan skor IoU per region
    all_wt_iou_scores, all_tc_iou_scores, all_et_iou_scores = [], [], []

    # Fungsi helper untuk kalkulasi metrik dari numpy array
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

    for i in tqdm(range(len(val_generator)), desc="Mengevaluasi data validasi"):
        test_image_batch, test_mask_batch = val_generator[i]
        test_pred_batch = my_model.predict(test_image_batch, verbose=0)
        
        test_mask_argmax = np.argmax(test_mask_batch, axis=4)
        test_pred_argmax = np.argmax(test_pred_batch, axis=4)
        
        # Hitung dice dan iou untuk semua kelas (0, 1, 2, 3)
        for cls in range(NUM_CLASSES):
            y_true_cls = (test_mask_argmax == cls)
            y_pred_cls = (test_pred_argmax == cls)
            all_class_dice_scores[cls].append(numpy_dice_score(y_true_cls, y_pred_cls))
            all_class_iou_scores[cls].append(numpy_iou_score(y_true_cls, y_pred_cls))
            
        # Hitung dice per region (WT, TC, ET)
        wt_true = np.isin(test_mask_argmax, [1, 2, 3]); wt_pred = np.isin(test_pred_argmax, [1, 2, 3])
        all_wt_scores.append(numpy_dice_score(wt_true, wt_pred))
        tc_true = np.isin(test_mask_argmax, [1, 3]); tc_pred = np.isin(test_pred_argmax, [1, 3])
        all_tc_scores.append(numpy_dice_score(tc_true, tc_pred))
        et_true = (test_mask_argmax == 3); et_pred = (test_pred_argmax == 3)
        all_et_scores.append(numpy_dice_score(et_true, et_pred))
        
        # Hitung IoU per region (WT, TC, ET) dengan mask yang sama
        all_wt_iou_scores.append(numpy_iou_score(wt_true, wt_pred))
        all_tc_iou_scores.append(numpy_iou_score(tc_true, tc_pred))
        all_et_iou_scores.append(numpy_iou_score(et_true, et_pred))

    print("\n\n===========================================================")
    print("--- HASIL EVALUASI RATA-RATA PADA SELURUH DATA VALIDASI ---")
    print("===========================================================")
    
    # Menampilkan rata-rata dice per kelas
    mean_dice_per_class = []
    print("\nEvaluasi Dice Score Rata-rata per Kelas:")
    for cls in range(NUM_CLASSES):
        mean_score = np.mean(all_class_dice_scores[cls])
        mean_dice_per_class.append(mean_score)
        print(f"  - Kelas {cls} ({CLASS_NAMES[cls]}) — Rata-rata Dice: {mean_score:.4f}")
    
    # Menampilkan rata-rata IoU per kelas
    mean_iou_per_class = []
    print("\nEvaluasi IoU Score Rata-rata per Kelas:")
    for cls in range(NUM_CLASSES):
        mean_score = np.mean(all_class_iou_scores[cls])
        mean_iou_per_class.append(mean_score)
        print(f"  - Kelas {cls} ({CLASS_NAMES[cls]}) — Rata-rata IoU: {mean_score:.4f}")
    
    # Menghitung dan menampilkan ringkasan utama
    mean_iou_overall = np.mean(mean_iou_per_class)
    weights_for_eval = class_weights[1:]
    weighted_avg_dice = np.sum(np.array(mean_dice_per_class[1:]) * weights_for_eval) / np.sum(weights_for_eval)
    
    print("\n-----------------------------------------------------------")
    print(f"Mean IoU (mIoU) Seluruh Kelas: {mean_iou_overall:.4f}")
    print(f"Rata-rata Dice Score Tertimbang (Kelas 1,2,3): {weighted_avg_dice:.4f}")
    print("-----------------------------------------------------------")
    
    # Menampilkan rata-rata dice per region (standar BraTS)
    print("\nEvaluasi Kinerja Rata-rata per Region (Standar BraTS):")
    print(f"  - Whole Tumor (WT)      — Rata-rata Dice: {np.mean(all_wt_scores):.4f}")
    print(f"  - Tumor Core (TC)       — Rata-rata Dice: {np.mean(all_tc_scores):.4f}")
    print(f"  - Enhancing Tumor (ET)  — Rata-rata Dice: {np.mean(all_et_scores):.4f}")
    print("===========================================================\n")
    
    # Menampilkan rata-rata IoU per region
    print(f"  - Whole Tumor (WT)      — Rata-rata IoU : {np.mean(all_wt_iou_scores):.4f}")
    print(f"  - Tumor Core (TC)       — Rata-rata IoU : {np.mean(all_tc_iou_scores):.4f}")
    print(f"  - Enhancing Tumor (ET)  — Rata-rata IoU : {np.mean(all_et_iou_scores):.4f}")
    print("===========================================================\n")

else:
    print(f"Model tidak ditemukan di {MODEL_SAVE_PATH}. Lewati evaluasi.")


#==============================================================================
# BAGIAN 7: (OPSIONAL) EVALUASI MODEL PADA SELURUH DATA TRAINING
#==============================================================================
print("\n--- Memulai Evaluasi Model pada Seluruh Data Training ---")

# Pastikan model sudah dimuat dari BAGIAN 6
if 'my_model' in locals() and my_model is not None:
    
    # Inisialisasi ulang list untuk menyimpan skor training
    all_class_dice_scores_train = [[] for _ in range(NUM_CLASSES)]
    all_class_iou_scores_train = [[] for _ in range(NUM_CLASSES)]
    all_wt_scores_train, all_tc_scores_train, all_et_scores_train = [], [], []
    all_wt_iou_scores_train, all_tc_iou_scores_train, all_et_iou_scores_train = [], [], []

    # Loop evaluasi menggunakan train_generator
    # VVVV UBAH DI SINI VVVV
    for i in tqdm(range(len(train_generator)), desc="Mengevaluasi data training"):
        train_image_batch, train_mask_batch = train_generator[i] # <-- UBAH DI SINI
        train_pred_batch = my_model.predict(train_image_batch, verbose=0) # <-- UBAH DI SINI
        
        train_mask_argmax = np.argmax(train_mask_batch, axis=4) # <-- UBAH DI SINI
        train_pred_argmax = np.argmax(train_pred_batch, axis=4) # <-- UBAH DI SINI
        
        # Hitung dice dan iou untuk semua kelas
        for cls in range(NUM_CLASSES):
            y_true_cls = (train_mask_argmax == cls) # <-- UBAH DI SINI
            y_pred_cls = (train_pred_argmax == cls) # <-- UBAH DI SINI
            all_class_dice_scores_train[cls].append(numpy_dice_score(y_true_cls, y_pred_cls))
            all_class_iou_scores_train[cls].append(numpy_iou_score(y_true_cls, y_pred_cls))
            
        # Hitung dice per region (WT, TC, ET)
        wt_true = np.isin(train_mask_argmax, [1, 2, 3]); wt_pred = np.isin(train_pred_argmax, [1, 2, 3]) # <-- UBAH DI SINI
        all_wt_scores_train.append(numpy_dice_score(wt_true, wt_pred))
        tc_true = np.isin(train_mask_argmax, [1, 3]); tc_pred = np.isin(train_pred_argmax, [1, 3]) # <-- UBAH DI SINI
        all_tc_scores_train.append(numpy_dice_score(tc_true, tc_pred))
        et_true = (train_mask_argmax == 3); et_pred = (train_pred_argmax == 3) # <-- UBAH DI SINI
        all_et_scores_train.append(numpy_dice_score(et_true, et_pred))
        
        # Hitung IoU per region (WT, TC, ET)
        all_wt_iou_scores_train.append(numpy_iou_score(wt_true, wt_pred))
        all_tc_iou_scores_train.append(numpy_iou_score(tc_true, tc_pred))
        all_et_iou_scores_train.append(numpy_iou_score(et_true, et_pred))

    print("\n\n===========================================================")
    print("--- HASIL EVALUASI RATA-RATA PADA SELURUH DATA TRAINING ---") # <-- UBAH DI SINI
    print("===========================================================")
    
    # Menampilkan rata-rata dice per kelas
    mean_dice_per_class_train = []
    print("\nEvaluasi Dice Score Rata-rata per Kelas (Training):") # <-- UBAH DI SINI
    for cls in range(NUM_CLASSES):
        mean_score = np.mean(all_class_dice_scores_train[cls])
        mean_dice_per_class_train.append(mean_score)
        print(f"  - Kelas {cls} ({CLASS_NAMES[cls]}) — Rata-rata Dice: {mean_score:.4f}")
    
    # Menampilkan rata-rata IoU per kelas
    mean_iou_per_class_train = []
    print("\nEvaluasi IoU Score Rata-rata per Kelas (Training):") # <-- UBAH DI SINI
    for cls in range(NUM_CLASSES):
        mean_score = np.mean(all_class_iou_scores_train[cls])
        mean_iou_per_class_train.append(mean_score)
        print(f"  - Kelas {cls} ({CLASS_NAMES[cls]}) — Rata-rata IoU: {mean_score:.4f}")

    # Menampilkan rata-rata dice & iou per region (standar BraTS)
    print("\nEvaluasi Kinerja Rata-rata per Region (Training):") # <-- UBAH DI SINI
    print(f"  - Whole Tumor (WT)      — Rata-rata Dice: {np.mean(all_wt_scores_train):.4f}")
    print(f"  - Tumor Core (TC)       — Rata-rata Dice: {np.mean(all_tc_scores_train):.4f}")
    print(f"  - Enhancing Tumor (ET)  — Rata-rata Dice: {np.mean(all_et_scores_train):.4f}")
    print(f"  - Whole Tumor (WT)      — Rata-rata IoU : {np.mean(all_wt_iou_scores_train):.4f}")
    print(f"  - Tumor Core (TC)       — Rata-rata IoU : {np.mean(all_tc_iou_scores_train):.4f}")
    print(f"  - Enhancing Tumor (ET)  — Rata-rata IoU : {np.mean(all_et_iou_scores_train):.4f}")
    print("===========================================================\n")
else:
    print("Model tidak ditemukan. Lewati evaluasi pada data training.")
    
# ==============================================================================
# BAGIAN 8: EVALUASI FINAL PADA DATA TESTING
# ==============================================================================
print("\n--- Memulai Evaluasi Final pada Seluruh Data Testing ---")

# Pastikan model sudah dimuat dari BAGIAN 6
if 'my_model' in locals() and my_model is not None:
    
    # Inisialisasi list untuk menyimpan skor testing
    all_class_dice_scores_test = [[] for _ in range(NUM_CLASSES)]
    all_class_iou_scores_test = [[] for _ in range(NUM_CLASSES)]
    all_wt_scores_test, all_tc_scores_test, all_et_scores_test = [], [], []
    all_wt_iou_scores_test, all_tc_iou_scores_test, all_et_iou_scores_test = [], [], []

    # Loop evaluasi menggunakan test_generator
    for i in tqdm(range(len(test_generator)), desc="Mengevaluasi data testing"):
        test_image_batch, test_mask_batch = test_generator[i]
        test_pred_batch = my_model.predict(test_image_batch, verbose=0)
        
        test_mask_argmax = np.argmax(test_mask_batch, axis=4)
        test_pred_argmax = np.argmax(test_pred_batch, axis=4)
        
        # Hitung dice dan iou untuk semua kelas
        for cls in range(NUM_CLASSES):
            y_true_cls = (test_mask_argmax == cls)
            y_pred_cls = (test_pred_argmax == cls)
            all_class_dice_scores_test[cls].append(numpy_dice_score(y_true_cls, y_pred_cls))
            all_class_iou_scores_test[cls].append(numpy_iou_score(y_true_cls, y_pred_cls))
            
        # Hitung dice per region (WT, TC, ET)
        wt_true = np.isin(test_mask_argmax, [1, 2, 3]); wt_pred = np.isin(test_pred_argmax, [1, 2, 3])
        all_wt_scores_test.append(numpy_dice_score(wt_true, wt_pred))
        tc_true = np.isin(test_mask_argmax, [1, 3]); tc_pred = np.isin(test_pred_argmax, [1, 3])
        all_tc_scores_test.append(numpy_dice_score(tc_true, tc_pred))
        et_true = (test_mask_argmax == 3); et_pred = (test_pred_argmax == 3)
        all_et_scores_test.append(numpy_dice_score(et_true, et_pred))
        
        # Hitung IoU per region (WT, TC, ET)
        all_wt_iou_scores_test.append(numpy_iou_score(wt_true, wt_pred))
        all_tc_iou_scores_test.append(numpy_iou_score(tc_true, tc_pred))
        all_et_iou_scores_test.append(numpy_iou_score(et_true, et_pred))

    print("\n\n===========================================================")
    print("--- HASIL EVALUASI FINAL PADA DATA TESTING ---")
    print("===========================================================")
    
    # Menampilkan rata-rata dice per kelas
    print("\nEvaluasi Dice Score Rata-rata per Kelas (Testing):")
    for cls in range(NUM_CLASSES):
        mean_score = np.mean(all_class_dice_scores_test[cls])
        print(f"  - Kelas {cls} ({CLASS_NAMES[cls]}) — Rata-rata Dice: {mean_score:.4f}")
    
    # Menampilkan rata-rata IoU per kelas
    print("\nEvaluasi IoU Score Rata-rata per Kelas (Testing):")
    for cls in range(NUM_CLASSES):
        mean_score = np.mean(all_class_iou_scores_test[cls])
        print(f"  - Kelas {cls} ({CLASS_NAMES[cls]}) — Rata-rata IoU: {mean_score:.4f}")

    # Menampilkan rata-rata dice & iou per region (standar BraTS)
    print("\nEvaluasi Kinerja Rata-rata per Region (Testing):")
    print(f"  - Whole Tumor (WT)      — Rata-rata Dice: {np.mean(all_wt_scores_test):.4f}")
    print(f"  - Tumor Core (TC)       — Rata-rata Dice: {np.mean(all_tc_scores_test):.4f}")
    print(f"  - Enhancing Tumor (ET)  — Rata-rata Dice: {np.mean(all_et_scores_test):.4f}")
    print(f"  - Whole Tumor (WT)      — Rata-rata IoU : {np.mean(all_wt_iou_scores_test):.4f}")
    print(f"  - Tumor Core (TC)       — Rata-rata IoU : {np.mean(all_tc_iou_scores_test):.4f}")
    print(f"  - Enhancing Tumor (ET)  — Rata-rata IoU : {np.mean(all_et_iou_scores_test):.4f}")
    print("===========================================================\n")
else:
    print("Model tidak ditemukan. Lewati evaluasi pada data testing.")
    
