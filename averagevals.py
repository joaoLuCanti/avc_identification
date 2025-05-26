from cnn_xception import create_generators, create_model

num_iteracoes = 5
epochs = 10

accuracies = []

for i in range(1, num_iteracoes + 1):
    train_generator, test_generator = create_generators()
    model = create_model()
    print(f"Iteração nº: {i}")

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        verbose=1
    )
    test_loss, test_acc, _, _ = model.evaluate(test_generator)
    accuracies.append(test_acc)

print("Média sem augmentation:")
print(sum(accuracies) / len(accuracies))