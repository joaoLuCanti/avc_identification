import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
from keras.metrics import Recall, Precision
from keras.applications import ResNet101, ResNet50, DenseNet121, EfficientNetB0
from keras.metrics import Accuracy, Recall, Precision
from keras.applications.resnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
img_height = 128
img_width = 128
batch_size = 64
epochs = 5  # 60

# Caminhos
train_dir = 'train'
test_dir = 'test'

# Geradores de dados
def create_datasets(use_augmentation=True):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False
    )

    # Augmentação condicional
    if use_augmentation:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.RandomContrast(0.1),
        ])
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    preprocess = lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
    train_ds = train_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, test_ds


train_ds, test_ds = create_datasets()

# Definição do modelo Xception
# base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))


def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(img_height, img_width, 3))
    new_input_layer = Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=(img_height, img_width, 1))
    base_model.layers[0] = new_input_layer
    base_model.trainable = True  # Congelar camadas do modelo base
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        # Conv2D(1024, (3, 3), strides=(2, 2), padding='same'),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy", Recall(), Precision()])

    return model


if __name__ == "__main__":
    model = create_model()
    # Treinamento do modelo
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds
    )

    # Avaliação do modelo
    test_loss, test_acc, _, _ = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc}")

    # Predições
    y_pred = model.predict(test_ds)
    y_pred_classes = np.round(y_pred).astype(int).reshape(-1)

    # Matriz de Confusão
    cm = confusion_matrix(test_ds.classes, y_pred_classes)
    print('Confusion Matrix')
    print(cm)

    # Plotando a Matriz de Confusão
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("conf_xception_non_ct.png")
    plt.show()

    # Relatório de Classificação
    print('Classification Report')
    target_names = list(test_ds.class_indices.keys())
    print(classification_report(test_ds.classes,
          y_pred_classes, target_names=target_names))

    # Plotando precisão e validação
    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy", color="red")
    plt.plot(history.history["val_accuracy"],
             label="val_accuracy", color="blue")
    plt.legend()
    plt.savefig("acc_val-acc_xception_non_ct.png")
    plt.show()

    # Salvando o modelo
    model.save('xception_model.h5')

    # Identificando entradas classificadas corretamente e incorretamente
    file_paths = test_ds.filepaths
    true_labels = test_ds.classes
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
