import os
from collections import defaultdict

# ler todas as imagens 0 e 1
test_dir = 'test'

images = []
for num in os.listdir(test_dir): # test/0 e test/1
    test_num_dir = os.path.join(test_dir, num)
    
    for image in os.listdir(test_num_dir): # imagens em test/0/ e test/1/
        image_path = os.path.join(test_num_dir, image)
        images.append(image_path)

# Contador de aproveitamento sem augmentation
contadorsemaugmentation = defaultdict(int)
# Contador de aproveitamento com augmentation
contadorcomaugmentation = defaultdict(int)

for i in range(1, 11):
    with open(f'correct_classified_sem_augmentation{i}.txt', 'r') as file:
        for line in file:
            line = line.strip()
            contadorsemaugmentation[line] += 1
    with open(f'correct_classified_com_augmentation{i}.txt', 'r') as file:
        for line in file:
            line = line.strip()
            contadorcomaugmentation[line] += 1

MAXVAL = 10

with open('resultadocontagem.txt', 'w+') as resultado:
    resultado.write('Aproveitamento de cada entrada sem e com augmentation:\n')
    for image in images:
        resultado.write('{:50}: Sem augmentation = {}%; Com augmentation = {}%;\n'.format(image, int(contadorsemaugmentation[image] / MAXVAL * 100), int(contadorcomaugmentation[image] / MAXVAL * 100)))