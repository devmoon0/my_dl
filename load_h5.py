import os
from imutils import paths
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt

import random

from loguru import logger
base_dir = os.path.join(os.getcwd(), 'logs')
config_dict = dict(
    rotation="00:00", ## 매일 12시에 새로운 로그 파일 생성
    format="[{time:YYYY-MM-DD HH:mm:ss}] | {level} | {message}",
    retention="7 days", ## 7일 후에 제거
    compression="tar",
    encoding="utf8"
)

logger.add(os.path.join(base_dir, 'myai_{time:YYYY-MM-DD}.log'), level="INFO", **config_dict)


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) #현재파일 경로
MODEL_DIR_PATH= os.path.join(CURRENT_PATH, "models") # 모델폴더 경로
LABLE_PATH= os.path.join(MODEL_DIR_PATH, "0705_vehicle_label.pickle") #라벨파일 경로
MODEL_PATH = os.path.join(MODEL_DIR_PATH,"test_0705_resnet152v2.h5")#모델파일경로
IMG_DATA_PATH = os.path.join(MODEL_DIR_PATH,"img")

RESULT_DAVE_PATH = os.path.join(MODEL_DIR_PATH,"result_save")


mlb = None
mlb = pd.read_pickle(LABLE_PATH)
labels = mlb.classes_ 
model = keras.models.load_model(MODEL_PATH)

IMAG_SHAPE=(224, 224)

def predict(show_log=False, show_img=False, save_result=False):
    ## 분류할 이미지데이터들을 가져옴
    test_image_paths = sorted(
        list(
            paths.list_images(IMG_DATA_PATH)
        )
    )
    # print(">>> test image path =", test_image_paths)

    id_list = []
    pred_list = []

    for image_path in test_image_paths:
        test_image = cv2.imread(image_path)
        test_image = cv2.resize(test_image, IMAG_SHAPE)
        # show_im = cv2.imread(image_path)
        # cv2_imshow(show_im)

        test_image = test_image.astype("float") / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        prediction = model.predict(test_image)
        proba = prediction[0]
        # print(np.round(proba, 3))
        idx = np.argmax(proba)

        df = pd.DataFrame({'pred':prediction[0]})
        df = df.sort_values(by='pred', ascending=False, na_position='first')
        accuracy = (df.iloc[0]['pred'])* 100
        id_list.append(image_path)
        pred_list.append(idx)
        if show_log:
            print(f">>> 예측률 : {accuracy:.2f}%")
            if accuracy <=89:
                logger.info(f"[{accuracy}]>>> file name = {image_path} / predict class = {mlb.classes_[idx]}({idx})")

            if mlb :
                print(f">>> file name = {image_path} / predict class = {mlb.classes_[idx]}({idx})")
            else:
                ### 라벨파일이 없는경우
                # Labels: ['bus' 'car' 'truck']
                class_dictionary = {'bus':0,'car':1, 'truck':2}
                for x in class_dictionary:
                    if class_dictionary[x] == (df[df == df.iloc[0]].index[0]):
                        print(f">>> file name = {image_path} / ### Class prediction = {x}")
                        break
                print("*"*10)
        if save_result:
            save_result(RESULT_DAVE_PATH, image_path ,idx)

    if show_img:
        res = pd.DataFrame({
            'id' : id_list,
            'label' : pred_list
            })
        display_image_grid(test_image_paths, res, {0:'bus',1:'car', 2:'truck'} )

def save_result(root, img_path, cls):
    """폴더에 저장하는 함수"""
    def makef(name):
        os.makedirs(name, exist_ok=True)

    result_img_path =os.path.join(root, "result")
    makef(result_img_path)
    # for l in labels:
    #     save_f = os.path.join(result_img_path, str(l))
    #     makef(save_f)
    vh1_path = os.path.join(result_img_path, "vh1")
    makef(vh1_path)
    vh2_path = os.path.join(result_img_path, "vh2")
    makef(vh2_path)
    vh3_path = os.path.join(result_img_path, "vh3")
    makef(vh3_path)

    if cls == 0:
        shutil.copy2(img_path, vh2_path)
    elif cls == 1:
        shutil.copy2(img_path, vh1_path)
    else:
        shutil.copy2(img_path, vh3_path)


def display_image_grid(images_filepaths, res, predicted_labels) :
    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 12), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        img = random.choice(res['id'].values)
        label = res.loc[res['id'] == img, 'label'].values[0]
        
        ax.imshow(plt.imread(img))
        ax.set_title(predicted_labels[label], fontsize = 15)
    plt.tight_layout()
    plt.savefig("./resutl.png")

if __name__ == '__main__':
    predict()