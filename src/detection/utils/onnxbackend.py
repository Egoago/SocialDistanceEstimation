import logging
import os
import warnings

import onnxruntime as ort

"""
To use GPU, onnxruntime-gpu must be installed instead of the regular onnxruntime.
A CUDA-capable GPU (or another form of hardware acceleration) is required.
ONNXBackend currently only supports CPU or CUDA acceleration.

See: "https://onnxruntime.ai/", and
    "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements"
"""

logger = logging.getLogger(__name__)


class ONNXBackend:
    # If `use_gpu` is None, it should conform to detector.py-s default behaviour:
    # """
    # If None, the default behaviour is chosen: use GPU if possible, otherwise use CPU.
    # """
    use_gpu = None

    @staticmethod
    def get_inference_session(file_name):
        assert os.path.isfile(file_name), f'{file_name} does not exist or is not a file'

        sess = ort.InferenceSession(file_name)

        if ONNXBackend.use_gpu is None:
            ONNXBackend.use_gpu = True if 'GPU' == ort.get_device() else False

        if ONNXBackend.use_gpu:
            # To use a gpu, onnxruntime-gpu has to be installed
            assert 'GPU' == ort.get_device(), 'GPU is not available or incorrect onnxruntime is used'
            sess.set_providers(['CUDAExecutionProvider'])
            logger.debug('Using GPU')
            assert 'GPU' == ort.get_device()
        else:
            if 'GPU' == ort.get_device():
                # onnxruntime-gpu is installed
                warnings.warn('GPU is available, but is not utilized')
            sess.set_providers(['CPUExecutionProvider'])
            logger.debug('Not using GPU')

        return sess
