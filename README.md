# Neural Style Transfer:
This code provides a TensorFlow implementation and pretrained models for **Artistic Neural Style Transfer**, as described in the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge.

The implementation is supported by Weights and Biases reports. The implementation is divided into two parts:

* [Part 1](https://app.wandb.ai/authors/nerual_style_transfer/reports/Part-1-Deep-representations-a-way-to-conceive-Neural-Style-Transfer--VmlldzoyMjQzNDY) - Deals with the visualisation of deep embeddings and the content representation of a pre-trained deep learning model.
* [Part 2](https://app.wandb.ai/authors/nerual_style_transfer/reports/Part-2-Deep-representations-a-way-to-conceive-Neural-Style-Transfer--VmlldzoyMjYyNzk) - Deals with the style representation and the Neural Style Transfer algorithm.

![nst.gif](https://github.com/ariG23498/NeuralStyleTransfer/tree/master/Assets/nst.gif)

## Folder Structure:

```bash
.
├── #1Content_Representation.ipynb      - For content representation
├── #2Amalgamation.ipynb                - For amalgamation
├── #3Style_Representation.ipynb        - For style representation
├── #4Style_Transfer.ipynb              - For style transfer
├── README.md
└── Utils                 
    ├── norm_colab.ipynb                - Usage of downloaded and normalizer
    └── vgg-norm.py                     - The normalizer in `tf.keras` 
```



## Data Used:

* ImageNet - For pre-training and normalization

Special Mention: We have used [ImageNet dataset downloader](https://github.com/mf1024/ImageNet-Datasets-Downloader) to download a specific amount of images from the ImageNet API.

## Weights and Models:

* [Normalised VGG16 weights](https://github.com/ariG23498/NeuralStyleTransfer/releases/tag/v1.0)
* [Unnormalised VGG16 weights](https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)

## Normalization:

[Normalization repository](https://github.com/corleypc/vgg-normalize) has been used to properly normalize the activation maps of the pre-trained VGG16 model. The code base of the repository has been modified a little bit to be able to harness `tf.keras`.

## Results

* **Content Representation**

  ![content.gif](https://github.com/ariG23498/NeuralStyleTransfer/tree/master/Assets/content.gif)

* **Amalgamation**

  ![amalgamation.gif](https://github.com/ariG23498/NeuralStyleTransfer/tree/master/Assets/amalgamation.gif)

* **Style Representation**

  ![style.gif](https://github.com/ariG23498/NeuralStyleTransfer/tree/master/Assets/style.gif)

* **Neural Style Transfer**

  ![nst.gif](https://github.com/ariG23498/NeuralStyleTransfer/tree/master/Assets/nst.gif)
  
* **Photo-Realistic Images**

  ![images.jpeg](https://github.com/ariG23498/NeuralStyleTransfer/tree/master/Assets/images.jpeg)

  ![style_images.jpeg](https://github.com/ariG23498/NeuralStyleTransfer/tree/master/Assets/style_images.jpeg)

## Report Authors

|         Name          |                    Github                    |                    Twitter                    |
| :-------------------: | :------------------------------------------: | :-------------------------------------------: |
| Aritra Roy Gosthipaty | [@ariG23498](https://github.com/ariG23498/)  |  [@ariG23498](https://twitter.com/ariG23498)  |
| Devjyoti Chakraborty  | [@cr0wley-zz](https://github.com/cr0wley-zz) | [@Cr0wley_zz](https://twitter.com/Cr0wley_zz) |