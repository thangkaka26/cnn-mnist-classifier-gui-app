import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.nn import softmax
from tensorflow.keras.models import load_model
from PIL import Image
from pathlib import Path


def get_resource_path(relative_path: str) -> Path:
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent.parent
    return base_path / relative_path


class CNN_MNIST_Classifier:
    def __init__(self):
        model_path = get_resource_path("saves/cnn_mnist_classifier.keras")
        self.__cnn_model = load_model(model_path)
        self.__img_path = None

    def set_image_path(self, image_path):
        f_path = Path(image_path)
        if f_path.exists():
            self.__img_path = f_path
        else:
            raise Exception("Invalid Image File !")

    def _img_to_mtx(self):
        try:
            mtx = Image.open(self.__img_path).convert('L')
            mtx = np.array(mtx).astype('float32').reshape(1,28,28,1)
            return mtx
        except:
            return None

    def _cnn_result(self):
        mtx = self._img_to_mtx()
        probs = self.__cnn_model.predict(mtx, verbose=0)
        pred = np.argmax(probs, axis=1)[0]
        
        return pred, probs

    def _postprocess_result(self):
        pred, probs = self._cnn_result()
        probs = (softmax(probs[0]).numpy().astype('float32') * 100).tolist()
        prob_dict = {}

        for i in range(10):
            prob_dict[str(i)] = round(probs[i], 1)

        items = list(prob_dict.items())
        midpoint = len(prob_dict) // 2
        
        prob_pt1 = dict(items[:midpoint])
        prob_pt2 = dict(items[midpoint:])

        return pred, prob_pt1, prob_pt2


    def show_result(self):
        try:
            img_mtx = self._img_to_mtx()
            pred, prob_pt1, prob_pt2 = self._postprocess_result()
            
            plt.figure(figsize=(5, 5))
            plt.title(
                f"CNN's result: {pred}",
                fontdict={'family':'monospace', 'size':18, 'weight':'bold'}
                )
            plt.imshow(
                img_mtx[0].squeeze(),
                cmap='binary_r',
                vmin=0, vmax=255
                )
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f"Probabilities (%)\n{str(prob_pt1)[1:-1]}\n{str(prob_pt2)[1:-1]}", fontdict={'family':'monospace', 'size':10, 'weight':'bold'})
            plt.show()
        except:
            plt.close()
            pass