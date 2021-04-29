# TISM
TensorFlow Image Segmentation Models

**Features** (*italic means available soon*)
* Build segmentation model with selectable architecture and encoder/decoder backbones
* Build **2D** and **3D** models
* 2 models architectures
  * UNet (skip concatenation)
  * LinkNet (skip addition)
* 3 backbones building blocks
  * VGG-like (basic convolutions with increasing number of filters each level)
  * ResNet-like (add the input to the output of the block)
  * *DenseNet-like*
* Create and use your own architecture or backbone easily


The interesting features of this library are the **3D** backbones and the possibility of creating your own backbone.
If you want to make 2D image segmentation, I recommend you the very cool
[qubvel/segmentation_models](https://github.com/qubvel/segmentation_models)
library which implement more architectures, and pretrained backbones from keras.applications.
