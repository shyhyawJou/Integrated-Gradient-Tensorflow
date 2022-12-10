from PIL import Image
import argparse
import numpy as np

import tensorflow as tf

from ig import Integrated_Gradient



def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='tensorflow model path (keras saved model folder)')
    parser.add_argument('-d', default='cpu', help='device ("cpu" or "cuda")')
    parser.add_argument('-img', help='img path')
    parser.add_argument('-step', default=20, type=int, help='number of steps in the path integral')
    return parser.parse_args()
    


def main():
    arg = get_arg()

    preprocess = tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224),
        tf.keras.layers.Rescaling(1 / 127.5, -1),
    ])
   
    device = arg.d
    if device == 'cuda' and len(tf.config.list_physical_devices('GPU')) == 0:
        raise ValueError('There is no cuda !!!')
    
    if arg.m is None:
        model = tf.keras.applications.MobileNetV2(classifier_activation=None)
    else:
        model = tf.keras.models.load_model(arg.m)
    model.summary()
        
    ig_obj = Integrated_Gradient(model, arg.d, preprocess, arg.step)
    
    print('\ndevice:', arg.d)
    print('img:', arg.img)
    
    img = Image.open(arg.img).convert('RGB')
    w, h = img.size
    img = tf.constant(np.array(img))
    
    # output is tf Tensor, heatmap is ndarray
    output, heatmap = ig_obj.get_heatmap(img, baseline=None)
    print('\nPredict label:', np.argmax(output, 1).item())
    
    img = Image.fromarray(img.numpy())
    heatmap = Image.fromarray(heatmap)
    result = Image.new('RGB', (2 * w, h))
    result.paste(img)
    result.paste(heatmap, (w, 0))
    result.show()
    heatmap.save('heatmap.jpg')


if __name__ == "__main__":
    main()
