import os
from imutils import paths
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) #현재파일 경로
MODEL_DIR_PATH= os.path.join(CURRENT_PATH, "models") # 모델폴더 경로
LABLE_PATH= os.path.join(MODEL_DIR_PATH, "0705_vehicle_label.pickle") #라벨파일 경로
MODEL_PATH = os.path.join(MODEL_DIR_PATH,"test_0705_DenseNet201.h5")#모델파일경로
IMG_DATA_PATH = os.path.join(CURRENT_PATH, "img")# 예측할 이미지데이터 폴더 경로

mlb = None
mlb = pd.read_pickle(LABLE_PATH)
labels = mlb .classes_ 
model = keras.models.load_model(MODEL_PATH)

IMAG_SHAPE=(224, 224)

def predict():
    ## 분류할 이미지데이터들을 가져옴
    test_image_paths = sorted(
        list(
            paths.list_images(IMG_DATA_PATH)
        )
    )
    # print(">>> test image path =", test_image_paths)

    for image_path in test_image_paths:
        test_image = cv2.imread(image_path)

        test_image = cv2.resize(
            test_image, IMAG_SHAPE
        )
        show_im = cv2.imread(image_path)
        # cv2_imshow(show_im)

        test_image = test_image.astype("float") / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        proba = model.predict(test_image)[0]
        # print(np.round(proba, 3))
        idx = np.argmax(proba)
        
        prediction = model.predict(test_image)
        df = pd.DataFrame({'pred':prediction[0]})
        df = df.sort_values(by='pred', ascending=False, na_position='first')
        # print(f">>> 예측률 : {(df.iloc[0]['pred'])* 100:.2f}%")
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

if __name__ == '__main__':
    predict()