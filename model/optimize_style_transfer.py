import logging
import cv2
import tensorflow as tf
import numpy as np

from tqdm import tqdm

logger = logging.getLogger(__name__)


class OptimizeStyleTransfer:
    def __init__(self,
                 learning_rate=0.001):
        self._lr = learning_rate

        self._optimizer = tf.keras.optimizers.Adam(self._lr)

        # build model
        pretrain_model = tf.keras.applications.VGG19(include_top=False)
        content_layer = ['block5_conv2']
        style_layer = ['block1_conv1',
                       'block2_conv1',
                       'block3_conv1',
                       'block4_conv1',
                       'block5_conv1']
        self._model = tf.keras.Model(inputs=pretrain_model.input,
                                     outputs=[pretrain_model.get_layer(name).output for name in
                                              style_layer + content_layer])
        self._num_content_layer = len(content_layer)
        self._num_style_layer = len(style_layer)

    @staticmethod
    def _gram_matrix(input_tensor):
        layer_sum = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
        size = tf.cast(tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)
        return layer_sum / size

    @staticmethod
    def _loss(predict_layers, target_layers):
        loss = []
        for predict_layer, target_layer in zip(predict_layers, target_layers):
            loss.append(
                tf.reduce_mean((target_layer - predict_layer) ** 2)
            )
        return tf.add_n(loss)

    def train(self, content_image, style_image, step):
        content_image = tf.expand_dims(
            tf.convert_to_tensor(tf.keras.applications.inception_v3.preprocess_input(content_image)), axis=0)
        style_image = tf.expand_dims(
            tf.convert_to_tensor(tf.keras.applications.inception_v3.preprocess_input(style_image)), axis=0)

        logger.debug("Image max: %s", tf.reduce_max(content_image).numpy())
        logger.debug("Image min: %s", tf.reduce_min(content_image).numpy())
        logger.debug("Image mean: %s", tf.reduce_mean(content_image).numpy())

        style_target = [self._gram_matrix(x) for x in self._model(style_image)[:self._num_style_layer]]
        content_target = self._model(content_image)[-self._num_content_layer:]

        logger.debug("Start train step.")

        transfer_image = tf.Variable(content_image, dtype=tf.float32)
        bar = tqdm(range(step))
        for i in bar:
            with tf.GradientTape() as tape:
                tape.watch(transfer_image)
                output = self._model(transfer_image)

                style_output = [self._gram_matrix(x) for x in output[:self._num_style_layer]]
                content_output = output[-self._num_content_layer:]

                loss = self._loss(style_output, style_target) + self._loss(content_output, content_target)

            gradients = tape.gradient(loss, transfer_image)
            self._optimizer.apply_gradients([(gradients, transfer_image)])

            self._show_tensor_image(transfer_image)
            bar.set_postfix({"loss": loss.numpy()})

        return ((tf.clip_by_value(transfer_image, -1, 1)[0] + 1) * 127.5).numpy().astype(np.uint8)

    @staticmethod
    def _show_tensor_image(tensor_image):
        tensor_image = tensor_image.numpy()
        if len(np.shape(tensor_image)) == 4:
            tensor_image = tensor_image[0]
        elif len(np.shape(tensor_image)) > 4 or len(np.shape(tensor_image)) < 3:
            raise ValueError

        image = (tensor_image * 0.5) + 0.5
        image = np.clip(image, 0, 1)
        cv2.imshow("tensor_image", image)
        cv2.waitKey(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    my_style_image = cv2.imread("../assets/style_image_1.jpg")
    my_style_image = cv2.resize(my_style_image, (500, 500))
    my_content_image = cv2.imread("../assets/content_image.jpg")
    my_content_image = cv2.resize(my_content_image, (500, 500))

    my_model = OptimizeStyleTransfer(0.02)
    gen_image = my_model.train(my_content_image, my_style_image, 1000)
    stack_result = np.concatenate([my_content_image, my_style_image, gen_image], axis=1)
    cv2.imshow("result", stack_result)
    cv2.waitKey(0)
