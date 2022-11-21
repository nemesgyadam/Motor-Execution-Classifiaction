import os
import threading
import cv2


stim_folder = "resources/stim_imgs"


class Stimulus:
    def __init__(self, classes, delay):
        self.load_config(classes, delay)

    def load_config(self, classes, delay):
        """
        Loads the stimulus images to memory.
        """
        self.images = {}
        for c in classes:
            self.images[c] = cv2.imread(os.path.join(stim_folder, c + ".png"))
        self.images["blank"] = cv2.imread(os.path.join(stim_folder, "blank.png"))
        self.delay = delay * 1000

    def show_img(self, task):
        """
        Actully show the stimulus.
        """
        cv2.destroyAllWindows()
        cv2.imshow("stim", self.images[task])
        cv2.setWindowProperty("stim", cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(self.delay)

    def show(self, task: str):
        """
        Start a thread to show the stimulus.
        """
        try:
            self.session_thread.join()
        except:
            ...
        self.session_thread = threading.Thread(
            target=self.show_img, args=(task,), daemon=True
        )
        self.session_thread.start()
