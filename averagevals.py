import tensorflow as tf
from cnn_xception_novo import create_datasets, create_model

num_iteracoes = 1
epochs = 1

# Sem augmentation
accuracies = []
curvasdeaprendizagem = []
for i in range(1, num_iteracoes + 1):
    train_ds, test_ds, _ = create_datasets(False)
    model = create_model()
    print(f"Iteração nº: {i}")

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        verbose=1
    )
    curvasdeaprendizagem.append(history.history['accuracy'])

    test_loss, test_acc, _, _ = model.evaluate(test_ds)
    accuracies.append(test_acc)

    del model
    del history
    K.clear_session()
    gc.collect()

plotCurvaDeAprendizagem(curvasdeaprendizagem, "Sem Augmentation")

withoutagumentation = sum(accuracies) / len(accuracies)
print("Média sem augmentation:")
print(withoutagumentation)

# Com augmentation
accuracies = []
curvasdeaprendizagem = []
for i in range(1, num_iteracoes + 1):
    train_ds, test_ds, _ = create_datasets()
    model = create_model()
    print(f"Iteração nº: {i}")

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        verbose=1
    )
    curvasdeaprendizagem.append(history.history['accuracy'])

    test_loss, test_acc, _, _ = model.evaluate(test_ds)
    accuracies.append(test_acc)

plotCurvaDeAprendizagem(curvasdeaprendizagem, "Com Augmentation")

withaugmentation = sum(accuracies) / len(accuracies)
print("Média com augmentation:")
print(withaugmentation)


print("Acurácia da média dos resultados:")
print(f"Média sem agumentation: {withoutagumentation}")
print(f"Média com augmentation: {withaugmentation}")
