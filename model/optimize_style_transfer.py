import cv2
import logging
import argparse
import tensorflow as tf
import numpy as np

from tqdm import tqdm

logger = logging.getLogger(__name__)


class OptimizeStyleTransfer:
    def __init__(self,
                 learning_rate=0.01,
                 model="VGG19",
                 content_layer=('block5_conv2',),
                 style_layer=('block1_conv1',
                              'block2_conv1',
                              'block3_conv1',
                              'block4_conv1',
                              'block5_conv1')):
        self._lr = learning_rate

        self._optimizer = tf.keras.optimizers.Adam(self._lr)

        # build model
        if model == "VGG19":
            pretrain_model = tf.keras.applications.VGG19(include_top=False)
        elif model == "InceptionV3":
            pretrain_model = tf.keras.applications.InceptionV3(include_top=False)
        elif isinstance(model, tf.keras.Model):
            pretrain_model = model
        else:
            raise ValueError

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

    def train(self, content_image, style_image, step, preview=False):
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

            if preview:
                self._show_tensor_image(transfer_image)
            bar.set_postfix({"loss": loss.numpy()})

        try:
            cv2.destroyWindow("tensor_image")
        except cv2.error:
            pass

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


def main(style_image_file, content_image_file, output_image_file,
         stack_output_image_file=None, step=100, resize=(500, 500), learning_rate=0.02):
    style_image = cv2.imread(style_image_file)
    style_image = cv2.resize(style_image, resize)
    content_image = cv2.imread(content_image_file)
    content_image = cv2.resize(content_image, resize)

    my_model = OptimizeStyleTransfer(learning_rate)

    gen_image = my_model.train(content_image, style_image, step)
    cv2.imwrite(output_image_file, gen_image)

    if stack_output_image_file is not None:
        stack_result = np.concatenate([content_image, style_image, gen_image], axis=1)
        cv2.imwrite(stack_output_image_file, stack_result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--style", "-s", type=str, help="Style image", required=True)
    parser.add_argument("--content", "-c", type=str, help="Content image", required=True)
    parser.add_argument("--output", "-o", type=str, help="Output image", required=True)
    parser.add_argument("--stack_output", type=str,
                        help="Save output image stack with style and content image for comparison")
    parser.add_argument("--resize", type=str, help="Resize image", default="500,500")
    parser.add_argument("--step", type=int, help="Train step", default=100)
    parser.add_argument("--lr", "--learning_rate", help="Learning rate", default=0.02)
    args = parser.parse_args()

    main(style_image_file=args.style,
         content_image_file=args.content,
         output_image_file=args.output,
         stack_output_image_file=args.stack_output,
         step=args.step,
         resize=tuple([int(x) for x in args.resize.split(',')]),
         learning_rate=args.lr)
    print("Done!")
