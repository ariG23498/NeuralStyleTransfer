import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import GeneratorEnqueuer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_folder = "/content/ImagNet/imagenet_images"  # REPLACE with a valid location
output_filename = "vgg16_norm_weights.h5"
batch_size = 64  # whatever your system can handle
img_size = (224, 224)  # best to set it to the size used in training your model
use_generator_enqueuer = True

def layer_mean_activations(x):
    """Filter activations averaged across positions and images in the batch"""
    if K.image_data_format() == 'channels_last':
        return K.mean(x, axis=(0, 1, 2))
    return K.mean(x, axis=(0, 2, 3))

if __name__ == "__main__":
    vgg_net = vgg16.VGG16(include_top=False, weights='imagenet')
    mean_activations = []
    x = vgg_net.input 
    for layer in vgg_net.layers[1:]: 
        x = layer(x)
        if isinstance(layer, Conv2D):
            mean_activations.append(layer_mean_activations(x))

    get_batch_means = K.function([vgg_net.input], mean_activations)

    idgen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
    gen = idgen.flow_from_directory(directory=data_folder,
                                    target_size=img_size, class_mode=None,
                                    shuffle=False, batch_size=batch_size)
    num_images = gen.samples

    if use_generator_enqueuer:
        enq = GeneratorEnqueuer(gen, use_multiprocessing=False)
        enq.start(workers=1)
        gen = enq.get()

    print("Gathering mean activations...")
    iters = num_images // batch_size
    accumulated_means = None
    for i in range(iters):
        batch_means = get_batch_means([next(gen)])
        if accumulated_means is None:
            accumulated_means = batch_means
        else:
            for accumulated, m in zip(accumulated_means, batch_means):
                accumulated += m
        if (i + 1) % 50 == 0: print("Batches done:", i + 1)

    for accumulated in accumulated_means:
        accumulated /= iters

    print("Normalizing mean activations...")
    means_iter = iter(accumulated_means)
    prev_conv_layer_means = None
    for layer in vgg_net.layers[1:]:
        if isinstance(layer, Conv2D):
            means = next(means_iter)
            W, b = layer.get_weights()
            # weights layout is: (dim1, dim2, in_channels, out_channels)
            if prev_conv_layer_means is not None:
                # undo upstream normalization to restore scale of incoming channels
                W *= prev_conv_layer_means[np.newaxis, np.newaxis, : , np.newaxis]
            # then normalize activations by rescaling both weights and biases
            b /= means
            W /= means[np.newaxis, np.newaxis, np.newaxis, :]
            layer.set_weights([W, b])
            prev_conv_layer_means = means
        
    vgg_net.save_weights(output_filename)

