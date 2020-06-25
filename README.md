# Hands-and-Face-Segmentation
Hands and Face Segmentation with Deep Convolutional Networks using limited labeled data.

## Usage

Training.

```python
from models import model1, model2
from keras import Model
import metrics
from keras.applications.vgg16 import VGG16

modelSNet = model1.SNet() # For using unet based architecture

# Using vgg based model.
modelVgg = VGG16(weights="imagenet", include_top=False, input_shape=(400,400,3))
base_model = Model(inputs=model.layers[0].output, outputs=model.layers[10].output)

# Freeze vgg layers.
for layer in base_model.layers:
    layer.trainable = False
    
modelVGG = model2.vggPre(base_model)

modelVGG.fit(...)
modelSNet.fit(...)



# Metrics for evaluting models, if you want to use pretrained model.
dependencies = {
	'f1_m' = metrics.f1_m,
	'recall_m' = metrics.recall_m,
	'precision_m' = metrics.precision_m
}



```

## DATASET

Ankara University Computer Vision & Machine Learning Labaratory (CVML LAB) Turkish Sign Language (TSL) Dataset.

For this study we used only a small portion ( 400 frames ) of the dataset which is consist of Turkish Sign Language videos (228 words ×∼ 150 samples ≈ 34.200 sample videos).You can access numpy version here: 

* Images: https://drive.google.com/file/d/10zZun4ruE8jQZFvX5Ec6TS7OFmwQW6K8/view?usp=sharing
* Labels: https://drive.google.com/file/d/1WAnuNv_mFntQ2yrz1l7lzddU5VP2TQOl/view?usp=sharing

( The whole dataset will be published soon at https://cvml.ankara.edu.tr/ )


## Models architectures

Model-1

![snet](https://user-images.githubusercontent.com/23141486/65591140-9c2f2500-df94-11e9-83bd-a02072e7ae88.jpg)

Model-2

![SegNetwork-VGG](https://user-images.githubusercontent.com/23141486/65586920-a7328700-df8d-11e9-9c79-009eba1c5592.jpg)

### Model Performances

Model-1's performance is better than pretrained-vgg model ( model-2).

![model_accuracy](https://user-images.githubusercontent.com/23141486/65591604-76565000-df95-11e9-8a3a-c504ae9ef07f.png)  ![model_loss](https://user-images.githubusercontent.com/23141486/65591610-79e9d700-df95-11e9-99f7-8e685fc32d88.png)



Note: Our research has been accepted by IEEE 3rd International Symposium on Multidisciplinary Studies and Innovative Technologies with paper title " Hand and Face Segmentation with Deep Convolutional Networks using Limited Labelled Data". https://ieeexplore.ieee.org/document/8932835.
