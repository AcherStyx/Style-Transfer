import tensorflow as tf
import cv2
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DeepDream:
    def __init__(self,
                 model="inception_v3",
                 output_conv_layer=("mixed3", "mixed5")):

        if model == "inception_v3":
            model = tf.keras.applications.InceptionV3(include_top=False)
            outputs = [model.get_layer(name).output for name in output_conv_layer]
            self.model = tf.keras.Model(inputs=model.inputs,
                                        outputs=outputs)
        else:
            raise ValueError

    def train(self, image, step, step_size, preview=False):
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)

        bar = tqdm(range(step))
        for _ in bar:
            with tf.GradientTape() as tape:
                loss = tf.constant(0.0)
                tape.watch(image)
                output = self.model(image)
                loss = tf.reduce_mean(output)

            gradients = tape.gradient(loss, image)
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            image += gradients * step_size
            image = tf.clip_by_value(image, -1, 1)
            if preview:
                cv2.imshow("model_train_gen",
                           ((image.numpy()[0] + 1.0) / 2.0 * 255).astype(np.uint8))
                cv2.waitKey(1)
            bar.set_postfix({"loss": loss.numpy()})
        return (image.numpy()[0] + 1.0) / 2

    def summary(self):
        self.model.summary()
        tf.keras.utils.plot_model(self.model, to_file="model.png", show_shapes=True, show_layer_names=True, dpi=50)


if __name__ == '__main__':
    my_model = DeepDream(output_conv_layer=["mixed1", "mixed2"])
    # my_model.summary()
    my_img = np.ones([400, 400, 3], dtype=np.uint8)
    gen_image = my_model.train(my_img, 100, 0.05)
    cv2.imshow("gen", gen_image)
    cv2.waitKey(0)
