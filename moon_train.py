#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## ref : https://github.com/yaxan/Naruto_Handsign_Classification/tree/411fc2f5c2caa363eddb175bbe4409e34d88f354
import os
import numpy as np
from imutils import paths
import cv2
import pickle

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended
from tensorflow.keras.utils import plot_model



CollectiveAllReduceExtended._enable_check_health = False

tf.compat.v1.disable_eager_execution()
PATH_TO_DATA= "/home/moon/workspace/oracle_python/vehicle"
PATH_TO_TRAIN_DATA = "D:\\code\\Naruto_Handsign_Classification\\train"
PATH_TO_TEST_DATA = "D:\\code\\Naruto_Handsign_Classification\\newTest"

BATCH_SIZE = 16
TRAIN_SIZE = 4777
TEST_SIZE = 300

IMG_SHAPE=(224,224,3)

def get_datagen(dataset, aug=False):
    """
    # ImageDataGenerator를 사용하여 데이터 증식 설정하기
    # datagen = ImageDataGenerator(
    #     rotation_range=20, # 이미지 회전 각도 범위
    #     width_shift_range=0.1, # 수평 방향으로 평행 이동 (전체 너비에 대한 비율)
    #     height_shift_range=0.1, # 수직 방향으로 평행 이동 (전체 높이에 대한 비율)
    #     shear_range = 0.1, # 전단(shearing transformation) 각도 범위
    #     zoom_range=0.1, # 이미지 확대 범위
    #     horizontal_flip = True, # 수평으로 뒤집기
    #     fill_mode='nearest') # 회전/가로•세로 이동에 의해 새롭게 생성해야할 픽셀을 채울 전략
    """
    if aug:
        datagen = ImageDataGenerator(
                            rescale=1./255,
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=25,
                            width_shift_range=0.3,
                            height_shift_range=0.3,
                            shear_range=0.5,
                            zoom_range=0.3,
                            horizontal_flip=False,
                            brightness_range=[0.8,1.1])
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
            dataset,
            target_size=(IMG_SHAPE[1], IMG_SHAPE[0]),
            color_mode='rgb',
            shuffle = True,
            class_mode='categorical',
            batch_size=BATCH_SIZE)

def _load_model(m, _include_top_=False, classes=3):
    """
    classes : 라벨갯수
    """
    if m == 'MN':
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=IMG_SHAPE, alpha=1.0, include_top=_include_top_, weights='imagenet', pooling=None)
    elif m =='RN':
        base_model= tf.keras.applications.ResNet50(include_top=_include_top_,
                        input_shape=IMG_SHAPE,
                        pooling='avg',classes=5,
                        weights='imagenet')
    elif m == 'VG':
        base_model = VGG16(include_top=_include_top_,
                        input_shape=IMG_SHAPE,
                        pooling='avg',classes=5,
                        weights='imagenet')
    elif m == 'IC':
        base_model = InceptionV3(include_top=_include_top_,
                        input_shape=IMG_SHAPE,
                        pooling='avg',classes=5,
                        weights='imagenet')
    for layer in base_model.layers[:-1]:
        layer.trainable = False

    top_model = Flatten()(base_model.output)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(classes, activation='softmax')(top_model) ##TODO: 라벨갯수로 변경

    model = Model(base_model.input, top_model, name='Altered_myModel')
    model.summary()
    plot_model(model, to_file='model.png')
    return model

def load_model(m):
    if m == 'MN':
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(224,224,3), alpha=1.0, include_top=False, weights='imagenet', pooling=None)
        
        for layer in model.layers[:-1]:
            layer.trainable = False

        mobile_net = Flatten()(model.output)
        mobile_net = Dropout(0.3)(mobile_net)
        mobile_net = Dense(4096, activation='relu')(mobile_net)
        mobile_net = Dropout(0.3)(mobile_net)
        mobile_net = Dense(1024, activation='relu')(mobile_net)
        mobile_net = Dropout(0.3)(mobile_net)
        mobile_net = Dense(12, activation='softmax')(mobile_net)

        mobile_net_mobile = Model(model.input, mobile_net, name='Altered_MobileNet')
        mobile_net_mobile.summary()

        model = mobile_net_mobile

    elif m =='RN':
        pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                        input_shape=(224,224,3),
                        pooling='avg',classes=5,
                        weights='imagenet')
        for layer in pretrained_model.layers[:-1]:
            layer.trainable=False

        resnet_model = Flatten()(pretrained_model.output)
        resnet_model = Dropout(0.3)(resnet_model)
        resnet_model = Dense(4096, activation='relu')(resnet_model)
        resnet_model = Dropout(0.3)(resnet_model)
        resnet_model = Dense(1024, activation='relu')(resnet_model)
        resnet_model = Dropout(0.3)(resnet_model)
        resnet_model = Dense(12, activation='softmax')(resnet_model)

        Altered_ResNet = Model(pretrained_model.input, resnet_model, name='Altered_MobileNet')

        model = Altered_ResNet

    elif m == 'VG':
        from keras.applications.vgg16 import VGG16

        pretrained_model = VGG16(include_top=False,
                        input_shape=(224,224,3),
                        pooling='avg',classes=5,
                        weights='imagenet')
        for layer in pretrained_model.layers[:-1]:
            layer.trainable=False

        vgg_model = Flatten()(pretrained_model.output)
        vgg_model = Dropout(0.3)(vgg_model)
        vgg_model = Dense(4096, activation='relu')(vgg_model)
        vgg_model = Dropout(0.3)(vgg_model)
        vgg_model = Dense(1024, activation='relu')(vgg_model)
        vgg_model = Dropout(0.3)(vgg_model)
        vgg_model = Dense(12, activation='softmax')(vgg_model)

        Altered_VGG = Model(pretrained_model.input, vgg_model, name='Altered_MobileNet')

        model = Altered_VGG

    elif m == 'IC':
        from keras.applications.inception_v3 import InceptionV3
        # load model
        pretrained_model = InceptionV3(include_top=False,
                        input_shape=(224,224,3),
                        pooling='avg',classes=5,
                        weights='imagenet')
        for layer in pretrained_model.layers[:-1]:
            layer.trainable=False

        incep_model = Flatten()(pretrained_model.output)
        incep_model = Dropout(0.3)(incep_model)
        incep_model = Dense(4096, activation='relu')(incep_model)
        incep_model = Dropout(0.3)(incep_model)
        incep_model = Dense(1024, activation='relu')(incep_model)
        incep_model = Dropout(0.3)(incep_model)
        incep_model = Dense(12, activation='softmax')(incep_model)

        Altered_Incep = Model(pretrained_model.input, incep_model, name='Altered_MobileNet')
        model = Altered_Incep

    return model


def get_zip_dataset(local_zip, save_path="."):
    """
    압축된 데이터셋 해제
    """
    import zipfile
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall(save_path)
    zip_ref.close()

def split_data(save_path):
    """
    데이터셋 학습/테스트 셋으로 분리
    """
    from tqdm import tqdm
    image_paths = sorted(
        list(paths.list_images(save_path))
    )
    print(">>> image count =", len(image_paths))
    
    images = []
    labels = []
    for image_path in tqdm(image_paths):
        try:
            image = cv2.imread(image_path)
            ## 이미지 크기가 각기 다르기 때문에, 크기를 균일하게 변경한다.
            image = cv2.resize(
                image, (IMG_SHAPE[1], IMG_SHAPE[0])
            )
            images.append(image)
            
            label = image_path.split(os.path.sep)[-2] ## TODO:레이블 생성을 위한 이미지 경로 분리
            labels.append([label])
        except Exception as e:
            print(e)
            continue

    print(">>> images count =", len(images))
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)

    ###  MultiLabelBinarizer 모듈을 사용하게 되면 알아서
    ### class의 종류(개수) 인식과 multi-label encoding을 수행한다.
    mlb = MultiLabelBinarizer()
    enc_labels = mlb.fit_transform(labels)
    # `MultiLabelBinarizer`를 디스크에 저장합니다
    print("[INFO] serializing label binarizer...")
    f = open("vehicle_label.pickle", "wb")
    f.write(pickle.dumps(mlb))
    f.close()

    print(">>> classes name =", mlb.classes_)
    from sklearn.model_selection import train_test_split

    seed = 47
    return train_test_split(
        images, enc_labels, test_size=0.2, random_state=seed
    )


if __name__ ==  '__main__':

    ## get_zip_dataset("/home/moon/workspace/oracle_python/car_img.zip",save_path=PATH_TO_DATA)
    x_train, x_test, y_train, y_test = split_data(PATH_TO_DATA)
    print(">> train test shape = {} {}".format(x_train.shape, y_train.shape))

    print(">> Start Model")
    model = _load_model('VG')
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',mode='max',factor=0.5, patience=10, min_lr=0.001, verbose=1)
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                            mode='auto', baseline=None, restore_best_weights=True)
    model.compile(loss='categorical_crossentropy',
                    optimizer=adam, metrics=['accuracy'])
    print(">> End Model")

    train_generator  = get_datagen(PATH_TO_DATA, True)
    # test_generator   = get_datagen(PATH_TO_TEST_DATA, False)

    history = model.fit(
        train_generator,
        validation_data=(x_test, y_test),#test_generator, 
        steps_per_epoch=len(x_train)// BATCH_SIZE,
        validation_steps=len(x_test)// BATCH_SIZE,
        shuffle=True,
        epochs=50,
        callbacks=[early_stopper],
        use_multiprocessing=False,
    )
    model.save("./vehicle.h5")
    print(">> END")
