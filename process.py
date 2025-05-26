import os
import numpy as np
import cv2
import pydicom as dy
import gdcm
import pylibjpeg
import copy

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)
os.makedirs("test/1", exist_ok=True)
os.makedirs("test/0", exist_ok=True)
os.makedirs("train/1", exist_ok=True)
os.makedirs("train/0", exist_ok=True)

def normalize_image(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

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
        #imagem_janelada = imagem_janelada - 1020
        imagem_janelada = normalize_image(imagem_janelada).astype(np.uint8)
        return imagem_janelada  
    #imagem_janelada = (255/np.max(imagem_janelada)*imagem_janelada).astype(np.uint8)  
    imagem_janelada = (255/np.max(imagem_janelada)*imagem_janelada).astype(np.uint8)
    return imagem_janelada

def test_image(img):
    for i in range(512):
        for j in range(512):
            cv2.circle(img, (i, j), 5, (0, 255, 0), 2)
            cv2.imshow("janela", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def split_image(img):
    marked_img = copy.copy(img)
    found_white = False
    count = 5
    point_found_left = 1
    point_found_right=1
    for i in range(512):
        if img[i, 256] == 255 and found_white == False:
            if found_white == False:
                found_white = True
                count = 5
                point_found_left = i
            else:
                count = count - 1
        else:
            found_white = False
            count=5

    for i in range(511, -1, -1):
        if img[i, 256] == 255 and found_white == False:
            if found_white == False:
                found_white = True
                count = 5
                point_found_right = i
            else:
                count = count - 1
        else:
            found_white = False
            count=5
        
    mid_point = int((point_found_right+point_found_left)/2)
    return mid_point

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    scale = 1.0  
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image
from sklearn.metrics import precision_score, recall_score, f1_score



def process_dycom():
    # angles_to_rotate = [15,30,45,60,75,90]
    angles_to_rotate = []
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
                    janelamento = [1020,1180]
                    CT_case = True
                    print(slices)
                    with open("CT.txt", 'a') as file:
                        file.write(slices+"\n")
                else:
                    with open("non_CT.txt", 'a') as file:
                        file.write(slices+"\n")
                    janelamento = [0, 150]
                    CT_case = False
                img = get_jan(img = img_dc, jan = janelamento, CT_case = CT_case)
                mid_point = split_image(img)
                # img[182:298,202:350] = 255
                if "test" in dataset:
                    new_dataset = "test"
                else:
                    new_dataset = "train"
                new_dataset_path = os.path.join(path,new_dataset)
                new_classification_path = os.path.join(new_dataset_path,classification)
                #test_image(img)
                img1 = img[:,mid_point:]
                img2 = img[:, 0:mid_point]
                img_path = os.path.join(new_classification_path, slices[:-4]+".png") 
                img1_path = os.path.join(new_classification_path, slices[:-4]+"_1"+".png") 
                img2_path = os.path.join(new_classification_path, slices[:-4]+"_2"+".png")
                hori_flip = cv2.flip(img, 1)
                img_hori_path = os.path.join(new_classification_path, slices[:-4]+"horizontal_flip"+".png")
                vert_flip = cv2.flip(img, 0)
                img_vert_path = os.path.join(new_classification_path, slices[:-4]+"vert_flip"+".png")
                img_equalized = cv2.equalizeHist(img)
                img_equalized_path = os.path.join(new_classification_path, slices[:-4]+"equalized"+".png")
                try:
                    cv2.imwrite(img_path, img)
                    # cv2.imwrite(img2_path, img1)
                    # cv2.imwrite(img2_path, img2)
                    #cv2.imwrite(img_hori_path, hori_flip)
                    #cv2.imwrite(img_vert_path, vert_flip)
                    for angle in angles_to_rotate:
                        img_rotated = rotate_image(img, angle)
                        img_rotate_path = os.path.join(new_classification_path, slices[:-4]+"rotated_"+f"{angle}"+".png")
                        cv2.imwrite(img_rotate_path, img_rotated)
                except Exception as e:
                    print(f"erro na imagem {img1_path}")
                    print(f"ERRO = {e}")
                #cv2.imwrite(img_equalized_path, img_equalized)

if __name__ == "__main__":
    process_dycom()
                
