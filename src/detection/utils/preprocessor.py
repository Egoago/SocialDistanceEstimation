from typing import NamedTuple, Tuple

import cv2
import numpy as np

OperationInfo = NamedTuple('OperationInfo', scale=float, pad_each_x=float, pad_each_y=float)


class Preprocessor:
    pad_value = 128.0

    @staticmethod
    def preprocess_image(im: np.ndarray, shape: Tuple[float, float, float, float]) \
            -> Tuple[np.ndarray, OperationInfo]:
        """
        Resizes the image to be not larger than the height and width of shape while keeping it's aspect ratio.
        The rest is filled with padding with value PreProcessor.pad_value.
        Then, the image is normalized to [0, 1).

        Only Height and Width of the shape are used.

        :param im: an image in
        :param shape: a tuple with Batches x Height x Width x Channels of the input of the model
        :return: The padded image and an OperationInfo describing the changes
        """
        im_height, im_width, im_channels = im.shape
        batch, height, width, channels = shape

        # Resize image
        scale = min(width / im_width, height / im_height)
        new_width, new_height = int(scale * im_width), int(scale * im_height)
        im_resized = cv2.resize(im, (new_width, new_height))

        # Pad image
        im_padded = np.full(shape=[height, width, 3], fill_value=Preprocessor.pad_value, dtype=np.float32)
        # float32 is needed
        pad_each_x, pad_each_y = (width - new_width) // 2, (height - new_height) // 2
        im_padded[pad_each_y:new_height + pad_each_y, pad_each_x:new_width + pad_each_x, :] = im_resized

        # Normalize image
        im_padded = im_padded / 255.

        return im_padded, OperationInfo(scale, pad_each_x, pad_each_y)
