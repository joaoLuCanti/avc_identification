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
from keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
img_height = 128
img_width = 128
batch_size = 64
epochs = 1  # 60

# Caminhos
train_dir = 'train'
test_dir = 'test'

def get_file_paths(directory, allowed_exts=('jpg', 'jpeg', 'png')):
    all_files = []
    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for ext in allowed_exts:
                all_files.extend(sorted(
                    tf.io.gfile.glob(f"{class_path}/*.{ext}")))
    return all_files

# Geradores de dados
def create_datasets(use_augmentation=True):
    # Carregar datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        batch_size=None,
        image_size=(img_height, img_width),
        shuffle=True
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        batch_size=None,
        image_size=(img_height, img_width),
        shuffle=False
    )

    class_names = test_ds.class_names

    # Augmentação condicional
    if use_augmentation:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.RandomContrast(0.1),
        ])
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    preprocess = lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
    train_ds = train_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    # Shuffle + batch + drop_remainder para garantir batches fixos no TPU
    train_ds = train_ds.shuffle(1000)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.cache()
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, test_ds, class_names


train_ds, test_ds, class_names = create_datasets()

def create_model():
    # Definição do modelo Xception
    # base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = True  # Congelar camadas do modelo base
    
    new_input_layer = Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=(img_height, img_width, 1))
    base_model.layers[0] = new_input_layer
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

def evaluate_model_results(model, test_ds, test_dir):
    # Extrair rótulos verdadeiros sem alteração
    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)

    # Predições
    y_pred = model.predict(test_ds)
    y_pred_classes = np.round(y_pred).reshape(-1)

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    # Plot matriz de confusão com labels 0 e 1 exatamente
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("conf_xception_non_ct.png")
    plt.show()

    # Relatório de classificação usando class_names só para o texto
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    # Obter lista de arquivos correspondentes
    file_paths = get_file_paths(test_dir)

    correct = [file_paths[i] for i in range(len(y_true)) if y_pred_classes[i] == y_true[i]]
    incorrect = [file_paths[i] for i in range(len(y_true)) if y_pred_classes[i] != y_true[i]]

    with open('correct_classified.txt', 'w') as f:
        for item in correct:
            f.write(f"{item}\n")

    with open('incorrect_classified.txt', 'w') as f:
        for item in incorrect:
            f.write(f"{item}\n")

    print(f"Correctly classified samples: {len(correct)}")
    print(f"Incorrectly classified samples: {len(incorrect)}")

    return y_true, y_pred_classes

if __name__ == "__main__":
    train_ds, test_ds, class_names = create_datasets()
    model = create_model()

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds
    )

    test_loss, test_acc, _, _ = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc}")

    y_true, y_pred_classes = evaluate_model_results(model, test_ds, test_dir)

    # Plotar precisão mantendo nome original do arquivo
    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy", color="red")
    plt.plot(history.history["val_accuracy"], label="val_accuracy", color="blue")
    plt.legend()
    plt.savefig("acc_val-acc_xception_non_ct.png")
    plt.show()

    model.save("xception_model.h5")