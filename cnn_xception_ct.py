import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.metrics import Recall, Precision

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




# Configurações
img_height = 71
img_width = 71
batch_size = 28
epochs = 20
description = "generic"

# Caminhos
train_dir = 'train'
test_dir = 'test'

# Geradores de dados
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Manter ordem para matriz de confusão
)

# Definição do modelo Xception
base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Congelar camadas do modelo base

base_model_name = base_model.name

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilação do modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Treinamento do modelo
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

# Predições
y_pred = model.predict(test_generator)
y_pred_classes = np.round(y_pred).astype(int).reshape(-1)

# Matriz de Confusão
cm = confusion_matrix(test_generator.classes, y_pred_classes)
print('Confusion Matrix')
print(cm)
  
# Plotando a Matriz de Confusão
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("conf_xception.png")
plt.show()

# Relatório de Classificação
print('Classification Report')
target_names = list(test_generator.class_indices.keys())
print(classification_report(test_generator.classes, y_pred_classes, target_names=target_names))

# Plotando precisão e validação
plt.figure()
plt.plot(history.history["accuracy"], label="accuracy", color="red")
plt.plot(history.history["val_accuracy"], label="val_accuracy", color="blue")
plt.legend()
plt.savefig("acc_val-acc_xception.png")
plt.show()

# Salvando o modelo
model.save('xception_model.h5')

# Identificando entradas classificadas corretamente e incorretamente
file_paths = test_generator.filepaths
true_labels = test_generator.classes
correct = []
incorrect = []

for i, (pred, true) in enumerate(zip(y_pred_classes, true_labels)):
    if pred == true:
        correct.append(file_paths[i])
    else:
        incorrect.append(file_paths[i])

print(f"Correctly classified samples: {len(correct)}")
print(f"Incorrectly classified samples: {len(incorrect)}")

# Salvando as listas de arquivos
with open('correct_classified.txt', 'w') as f:
    for item in correct:
        f.write("%s\n" % item)

with open('incorrect_classified.txt', 'w') as f:
    for item in incorrect:
        f.write("%s\n" % item)

from sklearn.metrics import precision_score, recall_score, f1_score

# Cálculo das métricas
precision = precision_score(true_labels, y_pred_classes)
recall = recall_score(true_labels, y_pred_classes)  # Sensibilidade
f1 = f1_score(true_labels, y_pred_classes)

# Salvando métricas em um arquivo .txt
with open('metrics_report.txt', 'w') as f:
    f.write(f"Description: {description}\n")
    f.write(f"Model: {base_model_name}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall (Sensitivity): {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
