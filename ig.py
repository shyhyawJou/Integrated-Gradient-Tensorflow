import cv2
import numpy as np
    
import tensorflow as tf



class Integrated_Gradient:
    def __init__(self, model, device, preprocess=None, steps=20):
        self.model = model
        self.device = 'GPU' if device == 'cuda' else 'CPU'
        self.prep = preprocess
        self.steps = steps
    
    def get_heatmap(self, img, baseline=None):
        with tf.device(self.device):
            # check whether the input shape is the same as baseline
            x = img[None] if self.prep is None else self.prep(img)[None]                
            if baseline is None:
                baseline = tf.zeros_like(x)
            self._check(x, baseline)                
            
            # get predict label
            output = self.model(x)
            pred_label = tf.argmax(output, 1).numpy().item()
            
            # compute integrated gradient
            with tf.GradientTape() as tape:
                x = tf.Variable(x)
                X, delta_X = self._get_X_and_delta(x, baseline, self.steps)
                grad = tape.gradient(tf.reduce_sum(self.model(X)[:, pred_label]), X)
                ig_grad = delta_X * (grad[:-1] + grad[1:]) / 2.          
                ig_grad = tf.reduce_sum(ig_grad, axis=0)

            # plot
            cam = tf.nn.relu(ig_grad)
            cam = cam / tf.reduce_max(cam) * 255.
            cam = cam.numpy().astype(np.uint8)
            cam = cv2.resize(cam, img.shape[:2][::-1])
            cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)[..., ::-1]       
                            
        return output, cam

    def _get_X_and_delta(self, x, baseline, steps):
        alphas = tf.cast(tf.linspace(0, 1, steps + 1), 'float32')
        alphas = tf.reshape(alphas, (-1, 1, 1, 1))
        delta_x = (x - baseline)
        x = baseline + alphas * delta_x
        return x, delta_x / steps
    
    def _check(self, x, baseline):
        if x.shape != baseline.shape:
            raise ValueError(f'input shape should equal to baseline shape. '
                             f'Got input shape: {x.shape}, '
                             f'baseline shape: {baseline.shape}') 

