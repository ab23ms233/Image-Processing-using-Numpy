# IMPORTING NECESSARY LIBRARIES
import numpy as np
from PIL import Image
from ImageProcessingUsingNumpy import ImageProcessing
from scipy.ndimage import convolve

def main():
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

    while True:     # Invalid Input / Rechoice
        print("1. Crop Image")
        print("2. Rotate Image")
        print("3. Flip Image")
        print("4. Apply Filter")
        print("5. Generate random square striped image")
        print()
        operation = int(input("Enter the number of the operation you want to perform: "))
        print()

        while True:     # Same Choice Loop
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
                result = img.crop_img(coord_2, coord_1)

            # Rotate Image
            elif operation == 2:
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
                result = img.rotate_img(num)

            # Flip Image
            elif operation == 3:
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
                        result = img.flip_img('h')
                        break
                    elif direction == 2:
                        print("Flipping image vertically...")
                        result = img.flip_img('v')
                        break
                    else:
                        print("Invalid choice. Please enter 1 or 2.")

            # Apply Filters
            elif operation == 4:
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
                        result = img.negative()
                        break
                    elif filter_choice == 2:
                        print("Grayscaling Image...")
                        result = img.grayscale()
                        break
                    elif filter_choice == 3:
                        print("Binarising Image...")
                        result = img.binarise()
                        break
                    elif filter_choice == 4:
                        print("Blurring Image...")
                        result = img.blur_img()
                        break
                    elif filter_choice == 5:
                        print("Sharpening Image...")
                        result = img.sharpen_img()
                        break
                    elif filter_choice == 6:
                        print("Detecting Edges...")
                        result = ImageProcessing.edge_detection(img.arr)
                        break
                    else:
                        print("Invalid choice. Choose again.")

            # Generate Image
            elif operation == 5:
                print("You chose to generate an image")
                print()
                total_px = int(input("Enter the length/breadth in pixles: "))
                divisions = int(input("Enter the number of divisions, i.e., number of stripes: "))
                merge = int(input("Enter the magnitude of merging pixels to be used to merge the divisions: "))
                channels = int(input("Enter the number of channels (i.e., 3 for RGB image. Must be one of 1, 3, 4): "))
                direction = input("Do you want horizontal or vertical stripes ('h' for horizontal, 'v' for vertical)?: ")
                print()

                result = ImageProcessing.generate_image(total_px, divisions, merge, channels, direction)    #type: ignore
             
            #Invalid Choice
            else:
                print("Invalid choice. Enter again")
                break

            ImageProcessing.display_img(result)     # Displaying the modified Image

            if operation != 5:      # No original image for image generation
                compare = input("Do you want to compare the original & modified images side-by-side (y/n)?: ")

                if compare.lower().strip() == 'y':  # If user wants to compare
                    print("Enter any text you want to be displayed with the original image.")
                    print("Enter x if you don't want any text.")
                    original_text = input()
                    print()

                    if original_text.lower().strip() == 'x':
                        original_text = ''

                    print("Enter any text you want to be displayed with the modified image.")
                    print("Enter x if you don't want any text.")
                    modified_text = input()
                    print()

                    if modified_text.lower().strip() == 'x':
                        modified_text = ''

                    ImageProcessing.compare_img((original_img, original_text), (result, modified_text))

            while True:     # Confirmation of whether user selects correctly for same_choice
                same_choice = input("Do you want to perform the same operation again (y/n)?: ")

                if same_choice.lower().strip() == 'y':
                    same_op = True
                    break
                elif same_choice.lower().strip() == 'n':
                    same_op = False
                    break
                else:
                    print("Invalid choice. Enter again")

            if not same_op:
                break   # Goes to All Operations
        

        



