import os
import numpy as np
import cv2
import pydicom as dy
import gdcm
import pylibjpeg
import copy

os.makedirs("train_ct", exist_ok=True)
os.makedirs("test_ct", exist_ok=True)
os.makedirs("test_ct/1", exist_ok=True)
os.makedirs("test_ct/0", exist_ok=True)
os.makedirs("train_ct/1", exist_ok=True)
os.makedirs("train_ct/0", exist_ok=True)

def get_jan(**kwargs):
    image = kwargs["img"]
    jan = kwargs["jan"]
    CT_case = False if "CT_case" not in kwargs else kwargs["CT_case"]
    
    # arraynp = image.copy().astype(np.float128)
    min_valor = min(jan)
    max_valor = max(jan)
    imagem_janelada = np.copy(image)
    imagem_janelada[image < min_valor] = min_valor
    imagem_janelada[image > max_valor] = max_valor
    if CT_case:
        imagem_janelada = imagem_janelada - 1000  
    imagem_janelada = 255/np.max(imagem_janelada)*imagem_janelada  
    return imagem_janelada

def split_image(img):
    marked_img = copy.copy(img)
    for i in range(256):
        print(img[256,i])
        if img[256,i] == 255:
            cv2.circle(marked_img, (256,i), 2, (255,255,0), 4)
            cv2.imshow("janela", cv2)
            cv2.waitKey(0)
            cv2.closeAllWindows()
    print("parou")
        
    

def process_dycom():
    path = os.getcwd()
    datasets =["test","train"]
    outro = ["1","0"]
    dycom_dataset = [x+"_dycom" for x in datasets]
    for dataset in dycom_dataset:
        dataset_path = os.path.join(path, dataset)
        for classification in os.listdir(dataset_path):
            classification_path = os.path.join(dataset_path,classification)
            for slices in os.listdir(classification_path):
                slices_path = os.path.join(classification_path,slices)
                img_dc = dy.dcmread(slices_path).pixel_array
                if "CT" in slices:
                    continue
                    janelamento = [1000,1150]
                    CT_case = True
                    print(slices)
                    with open("CT.txt", 'a') as file:
                        file.write(slices+"\n")
                else:
                    with open("non_CT.txt\n", 'a') as file:
                        file.write(slices+"\n")
                    janelamento = [0, 150]
                    CT_case = False
                img = get_jan(img = img_dc, jan = janelamento, CT_case = CT_case)
                split_image(img)
                # img[182:298,202:350] = 255
                if "test" in dataset:
                    new_dataset = "test_non_ct"
                else:
                    new_dataset = "train_non_ct"
                new_dataset_path = os.path.join(path,new_dataset)
                new_classification_path = os.path.join(new_dataset_path,classification)
                img_path = os.path.join(new_classification_path, slices[:-4]+".png")
                #cv2.imwrite(img_path, img)

if __name__ == "__main__":
    process_dycom()
                
