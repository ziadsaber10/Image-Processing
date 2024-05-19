#                                   Image Processing
#                                 Final Practical Exam
#Task to aplly all Filters in Image Processing
# Data_of_Team
# first__member:- Ziad Saber Abdelfatah Romia || Academic_number:- 15210132 || Level:- 2 || Sec:- 30
# second_member:- Ahmed Khaled Hamed Anter    || Academic_number:- 15210017 || Level:- 2 || Sec:- 27
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageFiltersApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filters App")
        self.root.geometry("800x600")

        self.image = None
        self.filtered_image = None

        # Create a frame for buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.upload_button = tk.Button(self.button_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(fill=tk.X, pady=5)

        self.filter_buttons = []
        self.filter_functions = [
            ("High-pass Filter", self.apply_hpf),
            ("Mean Filter", self.apply_mean_filter),
            ("Median Filter", self.apply_median_filter),
            ("Roberts Edge Detector", self.apply_roberts_edge_detector),
            ("Prewitt Edge Detector", self.apply_prewitt_edge_detector),
            ("Sobel Edge Detector", self.apply_sobel_edge_detector),
            ("Erosion", self.apply_erosion),
            ("Dilation", self.apply_dilation),
            ("Opening", self.apply_open),
            ("Closing", self.apply_close),
            ("Hough Circle Transform", self.apply_hough_circle_transform)
        ]

        for filter_name, filter_func in self.filter_functions:
            button = tk.Button(self.button_frame, text=filter_name, command=filter_func)
            button.pack(fill=tk.X, pady=2)
            self.filter_buttons.append(button)

        # Image display label
        self.image_label = tk.Label(root)
        self.image_label.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.filtered_image = self.image.copy()
            self.show_image(self.image, "Original Image")

    def show_image(self, img, window_title):
        """Display the image in a new Tkinter Toplevel window."""
        new_window = tk.Toplevel(self.root)
        new_window.title(window_title)

        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)

        label = tk.Label(new_window, image=imgtk)
        label.image = imgtk  # Keep a reference to avoid garbage collection
        label.pack()

    def apply_hpf(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        self.filtered_image = cv2.filter2D(self.image, -1, kernel)
        self.show_image(self.filtered_image, "High-pass Filter")

    def apply_mean_filter(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        self.filtered_image = cv2.blur(self.image, (5, 5))
        self.show_image(self.filtered_image, "Mean Filter")

    def apply_median_filter(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        self.filtered_image = cv2.medianBlur(self.image, 5)
        self.show_image(self.filtered_image, "Median Filter")

    def apply_roberts_edge_detector(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        kernelx = np.array([[1, 0], [0, -1]], dtype=int)
        kernely = np.array([[0, 1], [-1, 0]], dtype=int)
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        x = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(img_gray, cv2.CV_16S, kernely)
        abs_x = cv2.convertScaleAbs(x)
        abs_y = cv2.convertScaleAbs(y)
        self.filtered_image = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        self.show_image(self.filtered_image, "Roberts Edge Detector")

    def apply_prewitt_edge_detector(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        x = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(img_gray, cv2.CV_16S, kernely)
        abs_x = cv2.convertScaleAbs(x)
        abs_y = cv2.convertScaleAbs(y)
        self.filtered_image = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        self.show_image(self.filtered_image, "Prewitt Edge Detector")

    def apply_sobel_edge_detector(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        self.filtered_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        self.show_image(self.filtered_image, "Sobel Edge Detector")

    def apply_erosion(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        kernel = np.ones((5, 5), np.uint8)
        self.filtered_image = cv2.erode(self.image, kernel, iterations=1)
        self.show_image(self.filtered_image, "Erosion")

    def apply_dilation(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        kernel = np.ones((5, 5), np.uint8)
        self.filtered_image = cv2.dilate(self.image, kernel, iterations=1)
        self.show_image(self.filtered_image, "Dilation")

    def apply_open(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        kernel = np.ones((5, 5), np.uint8)
        self.filtered_image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        self.show_image(self.filtered_image, "Opening")

    def apply_close(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        kernel = np.ones((5, 5), np.uint8)
        self.filtered_image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        self.show_image(self.filtered_image, "Closing")

    def apply_hough_circle_transform(self):
        if self.image is None:
            print("Please upload an image first.")
            return
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0,
                                   maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(self.image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(self.image, (i[0], i[1]), 2, (0, 0, 255), 3)
        self.show_image(self.image, "Hough Circle Transform")


# Main application
root = tk.Tk()
app = ImageFiltersApp(root)
root.mainloop()
