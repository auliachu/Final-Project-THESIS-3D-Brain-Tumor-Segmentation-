# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 20:57:27 2025

@author: Aulia Salsabila
"""

import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_average_distribution(all_mask_paths, num_classes):
    """
    Menganalisis RATA-RATA persebaran kelas dari DAFTAR PATH LENGKAP.

    Args:
        all_mask_paths (list): Daftar path lengkap ke semua file mask.
        num_classes (int): Jumlah total kelas.
    """
    if not all_mask_paths:
        print("Error: Tidak ada file mask yang diberikan dalam daftar.")
        return

    all_image_counts = []
    print(f"\n--- Menganalisis Rata-rata untuk {len(all_mask_paths)} Gambar Gabungan ---")
    
    # Langsung loop melalui daftar path yang sudah diberikan
    for filepath in tqdm(all_mask_paths, desc="Processing All Masks"):
        try:
            loaded_mask = np.load(filepath)
            
            if loaded_mask.ndim == 4:
                mask_data = np.argmax(loaded_mask, axis=-1)
            else:
                mask_data = loaded_mask.astype(np.uint8)
            
            voxel_counts = np.zeros(num_classes, dtype=np.int64)
            class_ids, counts = np.unique(mask_data, return_counts=True)
            
            for class_id, count in zip(class_ids, counts):
                if class_id < num_classes:
                    voxel_counts[class_id] = count
            
            all_image_counts.append(voxel_counts)
        except Exception as e:
            print(f"Gagal memproses {os.path.basename(filepath)}: {e}")

    if not all_image_counts:
        print("Tidak ada data yang berhasil diproses.")
        return

    # Hitung rata-rata dari semua gambar
    average_counts = np.mean(all_image_counts, axis=0)
    print(average_counts.shape)
    
    print("\n--- Hasil Rata-rata Persebaran Kelas ---")
    print("Rata-rata Jumlah Voxel per Kelas di Seluruh Dataset:")
    for i in range(num_classes):
        print(f"  - Kelas {i}: {average_counts[i]:,.2f} voxel")

    # --- Visualisasi Rata-rata ---
    plt.figure(figsize=(12, 6))
    class_labels = [f'Kelas {i}' for i in range(num_classes)]
    
    # Plot 1: Skala Normal
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(class_labels, average_counts, color='skyblue')
    ax1.set_title('Rata-rata Persebaran Voxel (Skala Normal)')
    ax1.set_ylabel('Rata-rata Jumlah Voxel')
    ax1.tick_params(axis='x', rotation=45)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Plot 2: Skala Logaritmik
    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(class_labels, average_counts, color='lightgreen')
    ax2.set_yscale('log')
    ax2.set_title('Rata-rata Persebaran Voxel (Skala Log)')
    ax2.set_ylabel('Rata-rata Jumlah Voxel (log)')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def calculate_class_weights(all_mask_paths, num_classes):
    """
    Menghitung bobot kelas dari DAFTAR PATH LENGKAP.
    """
    if not all_mask_paths:
        print("Error: Tidak ada file mask yang diberikan dalam daftar.")
        return None

    # Inisialisasi penghitung total untuk seluruh dataset
    total_counts = np.zeros(num_classes, dtype=np.int64)
    print(f"\n--- Menghitung Bobot Kelas dari {len(all_mask_paths)} Gambar ---")

    for filepath in tqdm(all_mask_paths, desc="Calculating Weights"):
        try:
            loaded_mask = np.load(filepath)
            if loaded_mask.ndim == 4:
                mask_data = np.argmax(loaded_mask, axis=-1)
            else:
                mask_data = loaded_mask.astype(np.uint8)
            
            class_ids, counts = np.unique(mask_data, return_counts=True)
            for class_id, count in zip(class_ids, counts):
                if class_id < num_classes:
                    total_counts[class_id] += count
        except Exception as e:
            print(f"Gagal memproses {os.path.basename(filepath)}: {e}")

    # Hitung total semua voxel di dataset
    total_voxels_in_dataset = np.sum(total_counts)
    if total_voxels_in_dataset == 0:
        print("Error: Tidak ada voxel yang terhitung.")
        return None

    # Hitung bobot menggunakan metode inverse frequency
    weights = 1.0 / (total_counts / total_voxels_in_dataset + 1e-6)

    # Normalisasi bobot agar jumlahnya sama dengan 1.
    normalized_weights = weights / np.sum(weights)
    
    return normalized_weights

#Class weights updated 

import numpy as np
from tqdm import tqdm
import os

def calculate_class_weights_2(all_mask_paths, num_classes):
    """
    Menghitung bobot kelas dari DAFTAR PATH LENGKAP.
    """
    if not all_mask_paths:
        print("Error: Tidak ada file mask yang diberikan dalam daftar.")
        return None

    # Inisialisasi penghitung total untuk seluruh dataset
    total_counts = np.zeros(num_classes, dtype=np.int64)
    print(f"\n--- Menghitung Bobot Kelas dari {len(all_mask_paths)} Gambar ---")

    for filepath in tqdm(all_mask_paths, desc="Calculating Weights"):
        try:
            loaded_mask = np.load(filepath)
            if loaded_mask.ndim == 4:
                mask_data = np.argmax(loaded_mask, axis=-1)
            else:
                mask_data = loaded_mask.astype(np.uint8)
            
            class_ids, counts = np.unique(mask_data, return_counts=True)
            for class_id, count in zip(class_ids, counts):
                if class_id < num_classes:
                    total_counts[class_id] += count
        except Exception as e:
            print(f"Gagal memproses {os.path.basename(filepath)}: {e}")

    # --- Bagian yang ditambahkan untuk mencetak hasil ---
    print("\n--- Hasil Akumulasi Voxel Per Kelas ---")
    for i in range(num_classes):
        print(f"  - Kelas {i}: {total_counts[i]:,d} voxel")
    # --- Akhir bagian yang ditambahkan ---

    # Hitung total semua voxel di dataset
    total_voxels_in_dataset = np.sum(total_counts)
    if total_voxels_in_dataset == 0:
        print("Error: Tidak ada voxel yang terhitung.")
        return None

    # Hitung bobot menggunakan metode inverse frequency
    weights = 1.0 / (total_counts / total_voxels_in_dataset + 1e-6)
    # --- Bagian yang ditambahkan untuk mencetak bobot awal ---
    print("\n--- Bobot Awal (Pre-Normalisasi) ---")
    print(weights)
    # --- Akhir bagian yang ditambahkan ---

    # Normalisasi bobot agar jumlahnya sama dengan 1.
    normalized_weights = weights / np.sum(weights)
    
    return normalized_weights


# --- CARA MENGGUNAKAN ---
# --- Konfigurasi ---
NUM_CLASSES = 4 # Sesuaikan jika jumlah kelas Anda berbeda

# --- Tentukan Path untuk SEMUA Dataset ---
# Ganti dengan path Anda yang sebenarnya
BRATS_2021_DIR = "C:/Aulia/BraTS 2021/FullData/"


# --- Gabungkan semua path gambar dari semua folder ---
all_image_paths = []
# Data training 2020
all_image_paths.extend(glob.glob(os.path.join(BRATS_2021_DIR, 'train/images', '*.npy')))
# Data validasi 2020
all_image_paths.extend(glob.glob(os.path.join(BRATS_2021_DIR, 'val/images', '*.npy')))

# Buat path mask yang sesuai dari path gambar
all_mask_paths = [p.replace('images', 'masks').replace('image_', 'mask_') for p in all_image_paths]

# Pastikan ada file yang ditemukan
if all_mask_paths:
    # 1. Analisis rata-rata untuk seluruh dataset gabungan
    analyze_average_distribution(all_mask_paths, NUM_CLASSES)

    # 2. Hitung bobot untuk weighted loss dari dataset gabungan
    class_weights = calculate_class_weights_2(all_mask_paths, NUM_CLASSES)
    if class_weights is not None:
        print("\n--- Bobot Kelas untuk Weighted Loss (Rentang 0-1) ---")
        print("Bobot yang dihitung (jumlah total = 1):")
        for i, w in enumerate(class_weights):
            print(f"  - Bobot Kelas {i}: {w:.6f}")
        print("\nArray bobot untuk disalin:")
        print(np.array2string(class_weights, formatter={'float_kind':lambda x: "%.6f" % x}))
else:
    print("Tidak ada file mask yang ditemukan di path yang diberikan. Periksa kembali path Anda.")



