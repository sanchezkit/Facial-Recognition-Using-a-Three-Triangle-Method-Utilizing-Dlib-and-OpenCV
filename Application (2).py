import tkinter as tk
from PIL import Image, ImageTk
import glob
import os
import numpy as np
import dlib
import cv2
from imutils import face_utils
from Alignment import alignments
from bbrect import bb_to_rect
from parameters_computation import parameters
import joblib


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_camera(self):
        return self.cap


# Initialization
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('custom_landmark_ver4.dat')
model = joblib.load('svm_classifier_poly_ver3.dat')

camera = Camera()
cap = camera.get_camera()

root = tk.Tk()
root.winfo_toplevel().title("Four Point Model")
root.bind('<Escape>', lambda e: root.quit())

'''Icon dictionary
def dic_imgs():
    imgs = {}
    for i in glob.glob("icons/*.png"):
        pathfile = i
        i = os.path.basename(i)
        name = i.split(".")[0]
        imgs[name] = tk.PhotoImage(file=pathfile)
    return imgs


imgs = dic_imgs()
'''
toolbar = tk.Frame(root)
toolbar.pack(side=tk.TOP, fill=tk.X)
tk.TK_SILENCE_DEPRECATION = 1
''' With Icons
b0 = tk.Button(
    toolbar,
    relief=tk.FLAT,
    compound=tk.LEFT,
    text="New",
    command=lambda: clear_prediction(),
    image=imgs["refresh"])
b0.pack(side=tk.LEFT, padx=0, pady=0)

b1 = tk.Button(
    toolbar,
    relief=tk.FLAT,
    compound=tk.LEFT,
    text="Capture",
    command=lambda: run_recognition(),
    image=imgs["camera"])
b1.pack(side=tk.LEFT, padx=0, pady=0)

b2 = tk.Button(
    toolbar,
    text="Close",
    compound=tk.RIGHT,
    command=root.destroy,
    relief=tk.FLAT,
    image=imgs["exit"])
b2.pack(side=tk.RIGHT, padx=0, pady=0)
'''
# No icons
b0 = tk.Button(
    toolbar,
    relief=tk.FLAT,
    compound=tk.LEFT,
    text="New",
    command=lambda: clear_prediction())
b0.pack(side=tk.LEFT, padx=0, pady=0)

b1 = tk.Button(
    toolbar,
    relief=tk.FLAT,
    compound=tk.LEFT,
    text="Capture",
    command=lambda: run_recognition())
b1.pack(side=tk.LEFT, padx=0, pady=0)

b2 = tk.Button(
    toolbar,
    text="Close",
    compound=tk.RIGHT,
    command=root.destroy,
    relief=tk.FLAT)
b2.pack(side=tk.RIGHT, padx=0, pady=0)
# sidebar
sidebar = tk.Frame(root, width=200, bg='white', height=550, relief='sunken', borderwidth=2)
sidebar.pack(expand=True, fill='both', side='left', anchor='nw')
sidebar.pack_propagate(0)

# main content area
mainarea = tk.Frame(root, bg='#CCC', width=500, height=500)
mainarea.pack(expand=True, fill='both', side='right')

#video/image output
capturearea = tk.Label(mainarea)
capturearea.pack()

#Text
ovar = tk.StringVar()
ovar1 = tk.StringVar()
var = tk.StringVar()
var1 = tk.StringVar()
text0 = tk.Message(sidebar, text="Parameters", bg='white', width=200)
text0.pack(side=tk.TOP)
text0.pack_propagate(0)
text1 = tk.Message(sidebar, textvariable=var, bg='white', width=300)
text1.pack(side=tk.LEFT, anchor='nw')
text1.pack_propagate(0)
text2 = tk.Message(mainarea, textvariable=var1, bg='#CCC', width=500)
text2.pack(side=tk.TOP, anchor='n')
text2.pack_propagate(0)


def create_window():
    window = tk.Toplevel(root)
    window.winfo_toplevel().title("Output Image")
    # sidebar
    osidebar = tk.Frame(window, width=200, bg='white', height=550, relief='sunken', borderwidth=2)
    osidebar.pack(expand=True, fill='both', side='left', anchor='nw')
    osidebar.pack_propagate(0)

    # main content area
    omainarea = tk.Frame(window, bg='#CCC', width=500, height=500)
    omainarea.pack(expand=True, fill='both', side='right')

    file_path = 'cmp/user.jpg'
    image = Image.fromarray(cv2.imread(file_path))
    img = ImageTk.PhotoImage(image)
    panel = tk.Label(omainarea, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")

    otext0 = tk.Message(osidebar, text="Parameters", bg='white', width=200)
    otext0.pack(side=tk.TOP)
    otext0.pack_propagate(0)
    otext1 = tk.Message(osidebar, textvariable=ovar, bg='white', width=300)
    otext1.pack(side=tk.LEFT, anchor='nw')
    otext1.pack_propagate(0)
    otext2 = tk.Message(omainarea, textvariable=ovar1, bg='#CCC', width=500)
    otext2.pack(side=tk.BOTTOM, anchor='s')
    otext2.pack_propagate(0)

    with open('cmp/parameter_output.txt', 'r') as file:
        data = file.read()
        ovar.set(data)

    window.mainloop()


def clear_prediction():
    var1.set("")


def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
        rect = bb_to_rect(face)

        if len(face) > 0:
            shape1 = predictor(gray, rect.bb_to_rect_gen())
            shape = face_utils.shape_to_np(shape1)
            text = "{} face(s) found".format(len(face))
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

            for (x, y, w, h) in face:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for (xx, yy) in shape:
                    cv2.circle(frame, (xx, yy), 1, (0, 0, 255), 5)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            capturearea.imgtk = imgtk
            capturearea.configure(image=imgtk)
            capturearea.after(10, show_frame)
            parameter_output = get_parameter_output(shape)
            var.set(parameter_output)



def get_parameter_output(shape):
    params = parameters()
    results = params.compute(shape[2], shape[3], shape[0], shape[1])

    parameter_output = \
        "Lines \n" + \
        "AB: " + str(results[0][0]) + "mm\n" + \
        "AC: " + str(results[0][1]) + "mm\n" + \
        "AD: " + str(results[0][2]) + "mm\n" + \
        "BC: " + str(results[0][3]) + "mm\n" + \
        "BD: " + str(results[0][4]) + "mm\n" + \
        "CD: " + str(results[0][5]) + "mm\n" + \
        "\nAngles in Triangle ACD\n" + \
        "CAD: " + str(results[1][0]) + "\n" + \
        "ACD: " + str(results[1][1]) + "\n" + \
        "ADC: " + str(results[1][2]) + "\n" + \
        "\nAngles in Triangle BDC\n" + \
        "BCD: " + str(results[2][0]) + "\n" + \
        "CBD: " + str(results[2][1]) + "\n" + \
        "BDC: " + str(results[2][2]) + "\n" + \
        "\nAngles in Triangle ABD\n" + \
        "BAD: " + str(results[3][0]) + "\n" + \
        "ABD: " + str(results[3][1]) + "\n" + \
        "ADB: " + str(results[3][2]) + "\n" + \
        "\nAreas of Inner Triangles\n" + \
        "ACD: " + str(results[4][0]) + "sq.mm\n" + \
        "BCD: " + str(results[4][1]) + "sq.mm\n" + \
        "\nArea of Outer Triangle\n" + \
        "ADB: " + str(results[5]) + "sq.mm"
    return parameter_output


def run_recognition():
    ret, frame = cap.read()
    capture_face(frame)
    image = read_image()
    prediction = draw_prediction(image)

    prediction_text = "Prediction: " + str(np.argmax(prediction))
    var1.set(prediction_text)
    ovar1.set(prediction_text)
    create_window()


def capture_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    rect = bb_to_rect(face)

    shape1 = predictor(gray, rect.bb_to_rect_gen())
    shape = face_utils.shape_to_np(shape1)
    align = alignments(predictor=shape1, predictor_shape_np=shape, image=gray)
    al = align.align()

    cv2.imwrite('cmp/user.jpg', al)
    parameter_output = get_parameter_output(shape)
    text_file = open("cmp/parameter_output.txt", "wt")
    n = text_file.write(parameter_output)
    text_file.close()


def preprocess_technique_using_triangle(gray, shape1):
    nose_top_x = shape1.part(0).x
    nose_top_y = shape1.part(0).y

    nose_bottom_x = shape1.part(1).x
    nose_bottom_y = shape1.part(1).y

    left_eye_x = shape1.part(2).x
    left_eye_y = shape1.part(2).y

    right_eye_x = shape1.part(3).x
    right_eye_y = shape1.part(3).y

    point0 = [nose_top_x - 8, nose_top_y - 8]
    point1 = [nose_bottom_x - 5, nose_bottom_y - 5]
    point2 = [left_eye_x - 8, left_eye_y - 8]
    point3 = [right_eye_x - 8, right_eye_y - 8]

    # Region of interest AB, CB, DB
    roi_tri = np.array([point2, point1, point3])
    rect1 = cv2.boundingRect(roi_tri)
    xxx, yyy, www, hhh = rect1
    croped = gray[yyy:yyy + hhh, xxx:xxx + www]

    roi_tri = roi_tri - roi_tri.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [roi_tri], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.resize(cv2.bitwise_and(croped, croped, mask=mask), (366, 366), interpolation=cv2.INTER_LANCZOS4)
    return dst


def read_image():
    file_path = 'cmp/user.jpg'
    image = cv2.imread(file_path)
    height, width, _ = image.shape
    print(image.shape)
    rect = bb_to_rect([[0, 0, width, height]])
    shape1 = predictor(image, rect.bb_to_rect_gen())
    al_final = preprocess_technique_using_triangle(image, shape1)
    return al_final


def draw_prediction(image):
    al_final = np.reshape(image, (1, image.shape[0] * image.shape[1] * image.shape[2]))  # Reshape Aligned Image
    prediction = model.predict_proba(al_final)
    return prediction


if __name__ == '__main__':
    show_frame()
    root.mainloop()


