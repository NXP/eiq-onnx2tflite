#
# Copyright 2023 NXP
#
# License: LA_OPT_Online Code Hosting NXP_Software_License
# See the LICENSE for more details.
#

import wave
from typing import Union, Tuple, List

import numpy
import numpy as np
from PIL import Image


class DataLoader:

    @staticmethod
    def load_image(path: str, dtype: np.dtype = np.uint8) -> numpy.ndarray:
        """
        Load single image as numpy array

        :param path: Path to loaded image.
        :param dtype: Type of the returned image data. np.uint8 by default.
        :return: List with image data as numpy array. Includes batch dimension.
        """
        img = Image.open(path)
        img = np.asarray(img).astype(dtype)
        return img[np.newaxis, :]

    @staticmethod
    def load_audio(path: str, shape: Union[Tuple, List]):
        with wave.open(path) as f:
            data = f.readframes(8000)

            # Load data as float32
            data = np.frombuffer(data, dtype=np.int8).astype(np.float32)

            # Reshape to model input shape
            data = np.asarray([data for _ in range(256)]).reshape(shape)

        return data
