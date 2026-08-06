import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    def _pred_result(self):
        mtx = self._img_to_mtx()
        prob = self.__cnn_model.predict(mtx, verbose=0)
        pred = np.argmax(prob, axis=1)[0]
        
        return pred

    def show_result(self):
        try:
            img_mtx = self._img_to_mtx()
            plt.figure(figsize=(5, 5))
            plt.title(f"CNN's Result: {self._pred_result()}")
            sns.heatmap(img_mtx[0].squeeze(), xticklabels=False, yticklabels=False, cbar=False, vmin=0, vmax=255, cmap='binary_r')
            plt.show()
        except:
            plt.close()
            pass