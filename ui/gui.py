import sys
sys.dont_write_bytecode = True

import tkinter as tk
from tkinter import filedialog
from models.cnn_mnist_model import CNN_MNIST_Classifier

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CNN for MNIST images classification")
        self.root.geometry("640x480")
        self.root.resizable(True, True)

        self.cnn_model = CNN_MNIST_Classifier()

        self.exec_btn = tk.Button(
            self.root,
            text="Open an Image File",
            command=self._get_cnn_result,
            font=("Helvetica", 20, "bold italic"),
            relief='raised',
            activebackground="gray",
            bd=5,
            width=20, height=2)
        self.exec_btn.pack(padx=120, pady=140)

        self.root.mainloop()


    def _get_cnn_result(self):
        try:
            img_path = filedialog.askopenfilename(
                parent=self.root,
                title="Select an Image File",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.tga *.jfif *.webp *avif"),
                    ("JPEG", "*.jpg *.jpeg *.jpe *.jfif *.exif"), ("PNG", "*.png"), ("TGA", "*.tga"),
                    ("AV1 (AVIF)", "*.avif"), ("WebP", "*.webp")]
            )
            if (img_path):
                self.cnn_model.set_image_path(img_path)
                self.cnn_model.show_result()
        except:
            pass