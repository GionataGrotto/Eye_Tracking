import sys
import cv2
import numpy as np
import os
import shutil
import math
from pynput.mouse import Listener

# === GLOBAL VARIABLES ===
immagine = None
indexClick = 0
indexAcquisition = 0
totAcquisitions = 15

# === NORMALIZE COLORS ===
def normalize_color(input_image):
    min_val = input_image.min()
    max_val = input_image.max()
    normalized = (input_image - min_val) / (max_val - min_val)
    return normalized

# === EYE DETECTION FUNCTION ===
def get_eyes():
    global immagine, indexAcquisition

    if not webcam.isOpened():
        print("Webcam is not available.")
        sys.exit()

    ret, frame = webcam.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        webcam.release()
        listener.stop()
        sys.exit()

    frame = cv2.flip(frame, 1)
    immagine = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rectangles = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)

    if len(rectangles) != 2:
        return None

    x1, y1, w1, h1 = rectangles[0]
    x2, y2, w2, h2 = rectangles[1]

    eye1 = frame[y1:y1 + h1, x1:x1 + w1]
    eye2 = frame[y2:y2 + h2, x2:x2 + w2]

    eye1 = normalize_color(cv2.resize(eye1, (32, 32)))[10:-10, 5:-5]
    eye2 = normalize_color(cv2.resize(eye2, (32, 32)))[10:-10, 5:-5]

    eyes = [eye1, eye2] if x1 < x2 else [eye2, eye1]
    indexAcquisition += 1
    print(f"Acquisition #{indexAcquisition}")
    print(f"Eye centers: {x1}, {x2}")

    eye1_center = (x1 + w1 // 2, y1 + h1 // 2)
    eye2_center = (x2 + w2 // 2, y2 + h2 // 2)
    cv2.circle(immagine, eye1_center, 35, (255, 0, 0), 4)
    cv2.circle(immagine, eye2_center, 35, (0, 0, 255), 4)

    return (np.hstack(eyes) * 255).astype(np.uint8)

# === ON MOUSE CLICK ===
def on_click(x, y, button, pressed):
    global indexClick
    if pressed:

        if not webcam.isOpened():
            print("Webcam closed. Ignoring click.")
            return
        indexClick += 1
        print(f"Click #{indexClick} at ({x}, {y})")

        eyes = get_eyes()
        if eyes is not None:
            filename = os.path.join(root, f"{x}_{y}_Edo.jpeg")
            cv2.imwrite(filename, eyes)

            if indexAcquisition >= totAcquisitions:
                print("------- Acquisition Complete -------")
                webcam.release()
                listener.stop()
                cv2.imshow("Image", immagine)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                sys.exit()

# === SETUP WORKING DIRECTORY ===
def prepare_directory():
    global root
    root = input("Insert folder path:\n").strip()
    print(f"\nSelected folder: {root}")

    if os.path.isdir(root):
        while True:
            user_input = input("Directory already exists. Overwrite? (yes/quit): ").strip().lower()
            if user_input == "yes":
                shutil.rmtree(root)
                break
            elif user_input == "quit":
                sys.exit("Program exited.")
            else:
                print("Invalid input. Try again.")
    os.mkdir(root)

# === MAIN ===
if __name__ == "__main__":
    # Load classifier
    cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    if cascade.empty():
        print("Error loading cascade classifier.")
        sys.exit()

    # Setup directory
    prepare_directory()

    # Open webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Cannot open camera")
        sys.exit()

    # Listen for mouse clicks
    with Listener(on_click=on_click) as listener:
        listener.join()
