import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[1], True)
        tf.config.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)

# Ekstraksi data set dari file zip
zip_path = "R:/myPython/fruit_data.zip"
extract_path = "R:/myPython/fruit_data"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Tentukan path untuk data train dan data validasi
train_dir = os.path.join(extract_path, "train")
validation_dir = os.path.join(extract_path, "valid")

# Inisialisasi generator gambar untuk augmentasi data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Persiapkan data train dan data validasi
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load pre-trained DenseNet model tanpa lapisan klasifikasi
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Membuat lapisan klasifikasi baru di atas model DenseNet
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)

# Menggabungkan model basis DenseNet dengan lapisan klasifikasi baru
model = Model(inputs=base_model.input, outputs=predictions)

# Menyeting lapisan basis DenseNet menjadi tidak dapat dilatih
for layer in base_model.layers:
    layer.trainable = False

# Kompilasi model
model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])

# Definisi callback EarlyStopping untuk mencegah overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Melatih model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
    callbacks=[early_stopping]
)

# Tampilkan loss dan akurasi setiap epoch
for epoch, metrics in enumerate(history.history):
    print(f"Epoch {epoch+1}: {metrics}")

# Tampilkan grafik loss dan akurasi
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Uji model dengan random sample dari dataset
test_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False,
    class_mode='categorical'
)

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Tampilkan classification report dan confusion matrix
class_labels = list(test_generator.class_indices.keys())
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Tampilkan classification scatter graphic dengan klasifikasi masing-masing kelas
for class_label in class_labels:
    true_indices = np.where(test_generator.classes == test_generator.class_indices[class_label])[0]
    predicted_indices = np.where(predicted_classes == test_generator.class_indices[class_label])[0]

    # Pastikan jumlah data yang sama antara true_indices dan predicted_indices
    num_samples = min(len(true_indices), len(predicted_indices))
    true_indices = true_indices[:num_samples]
    predicted_indices = predicted_indices[:num_samples]

    plt.scatter(true_indices, predicted_indices, label=class_label)

plt.title('Classification Scatter Plot')
plt.xlabel('True Indices')
plt.ylabel('Predicted Indices')
plt.legend()
plt.show()

