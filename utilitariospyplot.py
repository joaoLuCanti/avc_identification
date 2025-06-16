import matplotlib.pyplot as plt
import numpy as np

def plotCurvaDeAprendizagem(accuracies, titulo = None):
    media_accuracies = np.mean(accuracies, axis=0)
    epocas = range(1, len(media_accuracies) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epocas, media_accuracies, label='Acurácia Média', color='blue', marker='o')

    plt.title(f'Curva de Aprendizagem Média {titulo if titulo else ""}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.grid(True)
    plt.legend()
    
    if titulo:
        plt.savefig(f"curva_aprendizagem_{titulo.replace(' ', '_').lower()}.png")
    plt.show()