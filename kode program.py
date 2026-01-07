import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import os
import math
import json

# --- 1. Definisi Path dan Parameter ---
# Sesuaikan path ini dengan lokasi Anda menyimpan dataset
train_path = 'Dataset_pemandangan/seg_train'
test_path = 'Dataset_pemandangan/seg_test'

# Parameter
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 20  
NUM_CLASSES = 6 # 'buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'

# --- 2. Pra-pemrosesan & Augmentasi Data ---
# Menggunakan ImageDataGenerator untuk memuat gambar dari direktori
# Kita juga menambahkan augmentasi sederhana pada data training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0
)

# Data validasi/test HANYA di-rescale, tidak diaugmentasi
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # Karena kita pakai categorical_crossentropy
)

validation_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False 
)

# --- 3. Membangun Model CNN ---
# Arsitektur diadaptasi dari kode asli, dibuat sedikit lebih dalam
model = Sequential()
# Blok 1
model.add(Conv2D(32, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Blok 2 (Mirip kode asli)
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Blok 3 (Tambahan untuk gambar lebih besar)
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Blok Fully Connected 
model.add(Flatten())
model.add(Dense(512)) # Ukuran layer disesuaikan
model.add(Activation("relu"))
model.add(Dropout(0.5)) # Menambah Dropout untuk mengurangi overfitting
model.add(Dense(NUM_CLASSES)) # Output layer dengan 6 kelas
model.add(Activation("softmax"))
model.summary()

# --- 4. Kompilasi Model ---
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001), # Menggunakan Adam
              metrics=['accuracy'])

# --- 5. Pelatihan Model ---
history = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=math.ceil(validation_generator.samples / BATCH_SIZE)
)

# --- 6. Evaluasi & Visualisasi Hasil ---

# Plot Akurasi
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'o-')
plt.plot(history.history['val_accuracy'], 'x-')
plt.title('Training/Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train acc', 'Validation acc'])

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'o-')
plt.plot(history.history['val_loss'], 'x-')
plt.title('Training/Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train loss', 'Validation loss'])
plt.show()

# --- 7. Confusion Matrix & Classification Report ---

# Mendapatkan prediksi
Y_pred = model.predict(validation_generator, validation_generator.samples // BATCH_SIZE + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Mendapatkan label asli
y_true = validation_generator.classes

# Membuat Confusion Matrix
print("\n--- Confusion Matrix ---")
confusion_mtx = confusion_matrix(y_true, y_pred)

# Visualisasi Confusion Matrix
class_names = list(train_generator.class_indices.keys())
df_cm = pd.DataFrame(confusion_mtx, index=class_names, columns=class_names)

plt.figure(figsize=(10, 8))
sn.set(font_scale=1.2) # Ukuran font
sn.heatmap(df_cm, cmap="Blues", annot=True, fmt='g', annot_kws={"size": 10})
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Laporan Klasifikasi
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# --- 8. save model ---
model_save_path = 'model_pemandangan_cnn.h5'
model.save(model_save_path)
print(f"Model berhasil disimpan ke: {model_save_path}")

class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
print("Class indices berhasil disimpan.")