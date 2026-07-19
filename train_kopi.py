import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. CEK HARDWARE
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ Memakai GPU Laptop")

# 2. DATASET (Arahkan ke folder 'train')
base_dir = 'train/' 

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    base_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training'
)
val_gen = datagen.flow_from_directory(
    base_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation'
)

class_names = list(train_gen.class_indices.keys())

# 3. ARSITEKTUR MODEL
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. MULAI TRAINING
print("\n🚀 Memulai Proses Training...")
model.fit(train_gen, validation_data=val_gen, epochs=50) 
model.save('model_kopi_v1.h5')
print("\n✅ SELESAI! Model disimpan sebagai model_kopi_v1.h5")