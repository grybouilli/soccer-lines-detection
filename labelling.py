import cv2
from pathlib import Path
from random import randint
from numpy import zeros
import numpy as np
import argparse

# Parse args
parser = argparse.ArgumentParser(description='Label images.')
parser.add_argument('--dir', type=str, help='Directory to label', required=True)

args = parser.parse_args()

# Variables to store line points
lines = []
history = []
current_line = []

# Callback function for mouse events
def draw_line(event, x, y, flags, param):
    global current_line, lines

    if event == cv2.EVENT_LBUTTONDOWN:
        current_line.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        current_line.append((x, y))
        lines.append(tuple(current_line))
        current_line = []

directory = args.dir

for file in Path("../data/" + directory).glob('*.png'):
    image_name = file.name

    img = cv2.imread(file.as_posix())
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_line)

    while True:
        # Display image
        temp_img = img.copy()
        
        mask_img = zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        for line in lines:
            cv2.line(temp_img, line[0], line[1], (0, 0, 255), 2)
            cv2.line(mask_img, line[0], line[1], (255, 255, 255), 10)

        cv2.imshow("Image", temp_img)

        key = cv2.waitKey(1) & 0xFF

        # Check for key presses
        if key == ord('s'):  # Save image
            result_path = "labels/" + directory + '/' + image_name
            cv2.imwrite(result_path, mask_img)
            print("Image saved.", result_path)
            lines = []
            break
        elif key == 27:  # ESC key
            lines = []
            break
        elif key == ord('z'):  # Undo last line
            if len(lines) > 0:
                history.append(lines.pop())
        elif key == ord('y'): # Redo last line
            if len(history) > 0:
                lines.append(history.pop())