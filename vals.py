import numpy as np
import tensorflow as tf
from cnn_xception import create_generators, create_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def printresultados(augmentation, test_generator, history):
    y_pred = model.predict(test_generator)
    y_pred_classes = np.round(y_pred).astype(int).reshape(-1)

    accuracy = accuracy_score(test_generator.classes, y_pred_classes)
    precision = precision_score(test_generator.classes, y_pred_classes)
    recall = recall_score(test_generator.classes, y_pred_classes)
    print(f'Accuracy: {accuracy * 100.0:.2f}%')
    print(f'Precision: {precision * 100.0:.2f}%')
    print(f'Recall: {recall * 100.0:.2f}%')

    # Matriz de Confusão
    cm = confusion_matrix(test_generator.classes, y_pred_classes)
    print("Confunsion Matrix:")
    print(cm)

    # Plotando a Matriz de Confusão
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    nome_plt_mc = f"conf_xception_non_ct_{'com' if augmentation else 'sem'}_augmentation.png"
    plt.savefig(nome_plt_mc)
    plt.show()

    # Relatório de Classificação
    print('Classification Report')
    target_names = list(test_generator.class_indices.keys())
    print(classification_report(test_generator.classes,
                                y_pred_classes, target_names=target_names))

    # Plotando precisão e validação
    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy", color="red")
    plt.plot(history.history["val_accuracy"],
             label="val_accuracy", color="blue")
    plt.legend()
    plt.savefig(
        f"acc_val-acc_xception_non_ct_{'com' if augmentation else 'sem'}_augmentation.png")
    plt.show()
    
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
    with open(f'correct_classified_{"com" if augmentation else "sem"}_augmentation.txt', 'w') as f:
        for item in correct:
            f.write("%s\n" % item)

    with open(f'incorrect_classified_{"com" if augmentation else "sem"}_augmentation.txt', 'w') as f:
        for item in incorrect:
            f.write("%s\n" % item)


epochs = 60
batch_size = 48

# Sem augmentation
train_generator, test_generator = create_generators(False)
model = create_model(batch_size)
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    verbose=1
)
test_loss, test_acc, _, _ = model.evaluate(test_generator)
printresultados(False, test_generator, history)

# Com augmentation
train_generator, test_generator = create_generators(True)
model = create_model(batch_size)
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    verbose=1
)
test_loss, test_acc, _, _ = model.evaluate(test_generator)
printresultados(True, test_generator, history)
