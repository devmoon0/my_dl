from tensorflow.keras.applications.vgg16 import VGG16                        
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19

model_VGG16 = VGG16(weights='imagenet')
model_VGG19 = VGG19(weights='imagenet')
model_RESNET50 = ResNet50(weights='imagenet')
model_INCEPTIONV3 = InceptionV3(weights='imagenet')