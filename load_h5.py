from imutils import paths
from tensorflow import keras
import cv2
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

mlb = pd.read_pickle(r'/home/moon/workspace/my_dl/vehicle_label.pickle')
labels = mlb .classes_
print(labels)
model = keras.models.load_model('/home/moon/workspace/my_dl/vehicle.h5')
# model.summary()

def predict():
    test_image_paths = sorted(
        list(
            paths.list_images("/home/moon/workspace/my_dl/img")
        )
    )
    print(">>> test image path =", test_image_paths)

    for image_path in test_image_paths:
        test_image = cv2.imread(image_path)

        test_image = cv2.resize(
            test_image, (32, 32)
        )
        show_im = cv2.imread(image_path)
        # cv2_imshow(show_im)

        test_image = test_image.astype("float") / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        proba = model.predict(test_image)[0]
        print(
            np.round(proba, 3)
        )
        idx = np.argmax(proba)
        print(idx)
        print(">>> predict class =", mlb.classes_[idx])

if __name__ == '__main__':
    predict()