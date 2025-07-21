# IMPORTING NECESSARY LIBRARIES
import numpy as np
from PIL import Image
from Class_ImgProcessing import ImageProcessing as ip
from colorama import Fore, init

def main():
    invalid_choice = Fore.RED + "Invalid Choice. Enter again"      # text to be displayed when user enters an invalid choice
    new_img = True      # User has to enter path of image
    init(autoreset=True)      # Initialize colorama to reset colors after each print

    # Introduction
    print()
    print(Fore.LIGHTYELLOW_EX + "Welcome to the Image Processing Program!")
    print()

    while True:     # flag = Invalid Input / All Operations
        print("Please choose an operation:")
        print("1. Crop Image")
        print("2. Rotate Image")
        print("3. Flip Image")
        print("4. Apply Filter")
        print("5. Generate random square striped image")
        print("6. Exit")
        print()
        operation = int(input("Enter choice: "))
        print()

        while True:     # flag = Same Choice

            if (operation != 5) and (new_img == True):      # No image required for image generation
                path = input("Enter the path to your image: ")

                # Converting Image to Numpy array
                img = Image.open(path)
                original_img = np.array(img)
                print(Fore.GREEN + "Your selected image is displayed")
                print()
                ip.display_img(original_img)

                # Creating an Instance of the ImageProcessing class
                img = ip(original_img)

            # Crop Image
            if operation == 1:
                print(Fore.CYAN + "You chose to crop the image.")
                print()
                x1 = int(input("Enter initial x coordinate: "))
                y1 = int(input("Enter initial y coordinate: "))
                x2 = int(input("Enter final x coordinate: "))
                y2 = int(input("Enter final y coordinate: "))
                print()
                print(Fore.GREEN + "Cropping the image...")
                print()

                coord_1, coord_2 = (x1, y1), (x2, y2)
                result = img.crop_img(coord_2, coord_1)

            # Rotate Image
            elif operation == 2:
                print(Fore.CYAN + "You chose to rotate the image.")
                print()
                print("Choose: ")
                print("1. Rotate clockwise")
                print("2. Rotate counter-clockwise")

                while True:     # flag = Invalid Rotate Direction
                    direction = int(input())
                    print()

                    if direction == 1:      # Clockwise
                        print(Fore.CYAN + "You chose to rotate clockwise.")
                        print("Available angles: 90, 180, 270")

                        while True:     # flag = Invalid Angle 1
                            angle = int(input("Enter the angle of rotation: "))
                            if angle in [90, 180, 270]:
                                break
                            else:
                                print(Fore.RED + "Invalid angle. Please enter 90, 180, or 270.")   # Returns to Invalid Angle 1

                        num = angle // 90   #Converting angle to number of turns
                        break

                    elif direction == 2:    # Counter-clockwise
                        print(Fore.CYAN + "You chose to rotate counter-clockwise.")
                        print("Available angles: 90, 180, 270")

                        while True:     # flag = Invalid Angle 2
                            angle = int(input("Enter the angle of rotation: "))
                            if angle in [90, 180, 270]:
                                break
                            else:
                                print(Fore.RED + "Invalid angle. Please enter 90, 180, or 270.")    # Returns to Invalid Angle 2

                        num = -angle // 90  #Converting angle to number of turns
                        break

                    else:   # Invalid Choice
                        print(invalid_choice)   # Returns to Invalid Rotate Direction

                print()
                print(Fore.GREEN + "Rotating the image...")
                print()
                result = img.rotate_img(num)

            # Flip Image
            elif operation == 3:
                print(Fore.CYAN + "You chose to flip the image.")
                print()
                print("Choose flip direction:")
                print("1. Horizontal")
                print("2. Vertical")

                while True:   # flag = Invalid Flip Direction
                    direction = int(input())
                    print()

                    if direction == 1:
                        print(Fore.GREEN + "Flipping image horizontally...")
                        print()
                        result = img.flip_img('h')
                        break
                    elif direction == 2:
                        print(Fore.GREEN + "Flipping image vertically...")
                        print()
                        result = img.flip_img('v')
                        break
                    else:
                        print(invalid_choice)   # Returns to Invalid Flip Direction

            # Apply Filters
            elif operation == 4:
                print(Fore.CYAN + "You chose to apply a filter to the image.")
                print()
                print("Available filters:")
                print("1. Negative")
                print("2. Grayscale")
                print("3. Binarise")
                print("4. Blur")
                print("5. Sharpen")
                print("6. Edge Detection")
                print()
                
                while True:     # flag = Invalid Filter Choice
                    filter_choice = int(input())
                    print()

                    if filter_choice == 1:
                        print(Fore.GREEN + "Generating Negative...")
                        print()
                        result = img.negative()
                        break
                    elif filter_choice == 2:
                        print(Fore.GREEN + "Grayscaling Image...")
                        print()
                        result = img.grayscale()
                        break
                    elif filter_choice == 3:
                        print(Fore.GREEN + "Binarising Image...")
                        print()
                        result = img.binarise()
                        break
                    elif filter_choice == 4:
                        print(Fore.GREEN + "Blurring Image...")
                        print()
                        result = img.blur_img()
                        break
                    elif filter_choice == 5:
                        print(Fore.GREEN + "Sharpening Image...")
                        print()
                        result = img.sharpen_img()
                        break
                    elif filter_choice == 6:
                        print(Fore.GREEN + "Detecting Edges...")
                        print()
                        result = img.edge_detection()
                        break
                    else:
                        print(invalid_choice)      # Returns to Invalid Filter Choice

            # Generate Image
            elif operation == 5:
                print(Fore.CYAN + "You chose to generate an image")
                print()
                total_px = int(input("Enter the length/breadth in pixles: "))
                divisions = int(input("Enter the number of divisions, i.e., number of stripes: "))
                merge = int(input("Enter the magnitude of merging pixels to be used to merge the divisions: "))
                channels = int(input("Enter the number of channels (i.e., 3 for RGB image. Must be one of 1, 3, 4): "))
                direction = input("Do you want horizontal or vertical stripes ('h' for horizontal, 'v' for vertical)?: ")
                print()
                print(Fore.GREEN + "Generating Image...")
                print()
                result = ip.generate_image(total_px, divisions, merge, channels, direction)    #type: ignore
            
            # Exit
            elif operation == 6:
                print(Fore.LIGHTYELLOW_EX + "Thank you for using the Image Processing Program!") 
                print()
                return  # Exits the program
            
            #Invalid Choice
            else:
                print(invalid_choice)
                break     # Returns to Invalid Input / All Operations

            ip.display_img(result)     # Displaying the modified Image

            if operation != 5:      # No original image for image generation, so no comparison
                compare = input(Fore.YELLOW + "Do you want to compare the original & modified images side-by-side (y/n)?: ")
                print()

                if compare.lower().strip() == 'y':  # If user wants to compare
                    print("Enter any text you want to be displayed with the original image.")
                    print("Enter x if you don't want any text.")
                    original_text = input()
                    print()

                    if original_text.lower().strip() == 'x':    # If user does not want any text
                        original_text = ''

                    print("Enter any text you want to be displayed with the modified image.")
                    print("Enter x if you don't want any text.")
                    modified_text = input()
                    print()

                    if modified_text.lower().strip() == 'x':    # If user does not want any text
                        modified_text = ''

                    ip.compare_img((original_img, original_text), (result, modified_text))     # Comaparing the images
            
            # What user wants to perform next
            print("OK. What do you want to do next?")
            print("1. Perform the same operation")
            print("2. Perform a different operation")
            print("3. Exit")

            while True:     # flag = Invalid choice
                choice = int(input())
                print()

                if choice == 1:
                    same_op = True     
                    break       # Goes to check_same_op -> All Operations
                elif choice == 2:
                    same_op = False
                    break       # Goes to check_same_op -> check_all_op
                elif choice == 3:
                    print(Fore.LIGHTYELLOW_EX + "Thank you for using the Image Processing Program!") 
                    print()
                    return  # Exits the program
                else:
                    print(invalid_choice)    #Goes to Invalid choice

            if operation != 5:  # Invalid options for generated image
                # Which Image user wants to operate on
                print("Which image do you want to operate on?")
                print("1. Original Image")
                print("2. New Image")
                print("3. Modified Image")
            else:   # Generating Images
                print("Which image do you want to operate on?")
                print("1. Generated Image")
                print("2. New Image")
                
            while True:     # flag = Invalid image_choice
                image_choice = int(input())
                print()

                if image_choice == 1:
                    new_img = False     # User does not have to enter path of image
                    if operation == 5:
                        img = ip(result)    # Instance of ImageProcessing class changes
                    else:
                        img = ip(original_img)    # Instance of ImageProcessing class changes
                    break 

                elif image_choice == 2:
                    new_img = True      # User has to enter path of image
                    break

                elif (image_choice == 3) and (operation != 5):
                    new_img = False    # Instance of ImageProcessing class remains the same
                    break

                else:
                    print(invalid_choice)   # Returns to flag Invalid image_choice
                
            # If same_op = True, the loop does not break and it returns to Same Choice
            # If same_op = False, it breaks and goes to All Operations
            if not same_op:     #flag = check_same_op
                break   # Goes to All Operations


if __name__ == "__main__":
    main()  # Calls the main function to run the program
        

        



