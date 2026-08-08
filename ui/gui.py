import sys
sys.dont_write_bytecode = True

import tkinter as tk
from tkinter import filedialog
from models.cnn_mnist_model import CNN_MNIST_Classifier
import webbrowser
from pathlib import Path


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CNN for MNIST images classification")
        self.root.geometry("640x480")
        self.root.resizable(True, True)

        self.cnn_model = CNN_MNIST_Classifier()
        self.src_btn_icon = tk.PhotoImage(file=Path(__file__).parent / "assets" / "github_icon.png")

        self.exec_btn = tk.Button(
            self.root,
            text="Open an Image File",
            command=self._get_cnn_result,
            font=("Helvetica", 20, "bold italic"),
            relief='raised',
            activebackground="gray",
            cursor="hand2",
            bd=5,
            width=20, height=2)
        self.exec_btn.pack(padx=120, pady=120)

        self.url_btn = tk.Button(
            self.root,
            text=" Source",
            command=lambda:webbrowser.open("https://github.com/thangkaka26/cnn-mnist-classifier"),
            image=self.src_btn_icon,
            compound="left",
            font=("Helvetica", 16, "bold"),
            activebackground="gray",
            relief='raised',
            cursor="hand2",
            bd=5,
            width=164, height=64
        )
        self.url_btn.image = self.src_btn_icon 
        self.url_btn.pack(padx=10, pady=0)

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