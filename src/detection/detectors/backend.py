import os

import onnxruntime as ort


class Backend:
    use_gpu = None

    @staticmethod
    def get_inference_session(file_name):
        assert os.path.isfile(file_name), f'{file_name} does not exist or is not a file'
        sess = ort.InferenceSession(file_name)
        if Backend.use_gpu is None:
            Backend.use_gpu = True if 'GPU' == ort.get_device() else False
        if Backend.use_gpu:
            assert 'GPU' == ort.get_device()
            sess.set_providers(['CUDAExecutionProvider'])
            assert 'GPU' == ort.get_device()
        else:
            if 'GPU' == ort.get_device():
                # onnxruntime-gpu is installed
                print('GPU is available, but is not utilized')
            sess.set_providers(['CPUExecutionProvider'])
        return sess
