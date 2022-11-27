import os
import time
import threading
from PIL import Image, ImageTk

from tkinter import Tk, Canvas, mainloop



root_path = "resources/stim_imgs"


class Stimulus:
    def __init__(self, stim_folder, classes, resize = False):
        self.stim_folder = stim_folder
        self.resize = resize
        self.start_GUI()
        self.load_images(classes)

    def load_images(self, classes):
        """
        Loads the stimulus images to memory.
        """
        self.images = {}

        for img_name in ["Blank", "Fixation"] + classes:
            pilImage = Image.open(
                os.path.join(root_path, self.stim_folder, img_name + ".jpg")
            )
            if self.resize:
                imgWidth, imgHeight = pilImage.size
                ratio = min(self.screenWidth / imgWidth, self.screenHeight / imgHeight)
                imgWidth = int(imgWidth * ratio)
                imgHeight = int(imgHeight * ratio)
                pilImage = pilImage.resize((imgWidth, imgHeight))
            self.images[img_name] = ImageTk.PhotoImage(pilImage)

    def tkinter_thread(self):
        """
        Thread responsible for tkinter window
        Note: The actual loop is created by mainloop tkinter function
        """
        self.root = Tk()
        self.root.bind("<Escape>", lambda x: self.root.destroy())

        self.screenWidth, self.screenHeight = (
            self.root.winfo_screenwidth(),
            self.root.winfo_screenheight(),
        )
        self.root.overrideredirect(1)
        self.root.geometry("%dx%d+0+0" % (self.screenWidth, self.screenHeight))
        self.root.focus_set()

        self.canvas = Canvas(
            self.root, width=self.screenWidth, height=self.screenHeight
        )
        self.canvas.pack()
        self.canvas.configure(background="white")
        self.canvas.pack()
        mainloop()

    def start_GUI(self):
        """
        Configure and start a main tkinter thread
        Note: Tkinter doesn't like to run a separated thead.
        """
        self.t1 = threading.Thread(target=self.tkinter_thread)
        self.t1.start()


    def show(self, task: str):
        """
        Change the canvas to the given stimulus image
        """
        self.canvas.delete("all")

        self.canvas.create_image(
            self.screenWidth / 2, self.screenHeight / 2, image=self.images[task]
        )

    def stop(self):
        """
        Stop every thead and close window.
        """
        self.root.quit()
        try:
            self.root.destroy()
        except:
            ...
        self.t1.join()
