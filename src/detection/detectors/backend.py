import os

import onnxruntime as ort


class Backend:
    use_gpu = False

    @staticmethod
    def get_inference_session(file_name):
        assert os.path.isfile(file_name), file_name
        sess = ort.InferenceSession(file_name)
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
