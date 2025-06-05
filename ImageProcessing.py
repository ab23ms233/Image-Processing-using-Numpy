
# IMPORTING NECESSARY LIBRARIES
import numpy as np
from PIL import Image
from ImageProcessingUsingNumpy import ImageProcessing
from scipy.ndimage import convolve

# Converting Image to Numpy array
img = Image.open('dog.jpg')
original_img = np.array(img)

# Creating an Instance of the ImageProcessing class
original_img_copy = original_img.copy()
img = ImageProcessing(original_img_copy)

# Introduction
print("Welcome to the Image Processing Program!")
print("Please choose an operation:")
print()

while True:
    print("1. Crop Image")
    print("2. Rotate Image")
    print("3. Flip Image")
    print("4. Apply Filter")
    print()

    while True: #Invalid input loop
        operation = int(input("Enter the number of the operation you want to perform: "))
        print()

        # Crop Image
        if operation == 1:
            print("You chose to crop the image.")
            print()
            x1 = int(input("Enter initial x coordinate: "))
            y1 = int(input("Enter initial y coordinate: "))
            x2 = int(input("Enter final x coordinate: "))
            y2 = int(input("Enter final y coordinate: "))
            print()
            print("Cropping the image...")

            coord_1, coord_2 = (x1, y1), (x2, y2)
            img.crop_img(coord_2, coord_1)
            break

        # Rotate Image
        if operation == 2:
            print("You chose to rotate the image.")
            print()
            print("Choose: ")
            print("1. Rotate clockwise")
            print("2. Rotate counter-clockwise")
            direction = int(input())
            print()
            print("Available angles: 90, 180, 270")

            if direction == 1:
                print("You chose to rotate clockwise.")
                print()

                while True:
                    angle = int(input("Enter the angle of rotation: "))
                    if angle in [90, 180, 270]:
                        break
                    else:
                        print("Invalid angle. Please enter 90, 180, or 270.")

                num = angle // 90   #Converting angle to number of turns

            elif direction == 2:
                print("You chose to rotate counter-clockwise.")
                print()

                while True:
                    angle = int(input("Enter the angle of rotation: "))
                    if angle in [90, 180, 270]:
                        break
                    else:
                        print("Invalid angle. Please enter 90, 180, or 270.")

                num = -angle // 90  #Converting angle to number of turns

            print()
            print("Rotating the image...")
            img.rotate_img(num)
            break

        # Flip Image
        if operation == 3:
            print("You chose to flip the image.")
            print()
            print("Choose flip direction:")
            print("1. Horizontal")
            print("2. Vertical")

            while True:
                direction = int(input())
                print()

                if direction == 1:
                    print("Flipping image horizontally...")
                    img.flip_img('h')
                    break
                elif direction == 2:
                    print("Flipping image vertically...")
                    img.flip_img('v')
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            break

        # Apply Filters
        if operation == 4:
            print("You chose to apply a filter to the image.")
            print()
            print("Available filters:")
            print("1. Negative")
            print("2. Grayscale")
            print("3. Binarise")
            print("4. Blur")
            print("5. Sharpen")
            print("6. Edge Detection")

            while True:
                filter_choice = int(input())
                print()

                if filter_choice == 1:
                    print("Generating Negative...")
                    img.negative()
                    break
                elif filter_choice == 2:
                    print("Grayscaling Image...")
                    img.grayscale()
                elif filter_choice == 3:
                    print("Binarising Image...")
                    img.binarise()
                elif filter_choice == 4:
                    print("Blurring Image...")
                    img.blur_img()
                    break
                elif filter_choice == 5:
                    print("Sharpening Image...")
                    img.sharpen_img()
                    break
                elif filter_choice == 6:
                    print("Detecting Edges...")
                    result = ImageProcessing.edge_detection(img.arr)
                    break
                else:
                    print("Invalid choice. Choose again.")
            break

        #Invalid Choice
        else:
            print("Invalid choice. Enter again")
        
    compare = input("Do you want to compare the original & modified images side-by-side (y/n)?: ")

    if compare.lower().strip() == 'y':
        original_text = input("Any text you want to be displayed with the original image (y/n)?")

        if text.lower().strip() == 'y':
            original_text = input


