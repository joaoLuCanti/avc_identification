from cnn_xception import create_datasets, create_model

num_iteracoes = 5
epochs = 6

# Sem augmentation
accuracies = []
for i in range(1, num_iteracoes + 1):
    train_ds, test_ds = create_datasets(False)
    model = create_model()
    print(f"Iteração nº: {i}")

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        verbose=1
    )
    test_loss, test_acc, _, _ = model.evaluate(test_ds)
    accuracies.append(test_acc)

print("Média sem augmentation:")
print(sum(accuracies) / len(accuracies))

# Com augmentation
accuracies = []
for i in range(1, num_iteracoes + 1):
    train_ds, test_ds = create_datasets()
    model = create_model()
    print(f"Iteração nº: {i}")

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        verbose=1
    )
    test_loss, test_acc, _, _ = model.evaluate(test_ds)
    accuracies.append(test_acc)
print("Média com augmentation:")
print(sum(accuracies) / len(accuracies))
