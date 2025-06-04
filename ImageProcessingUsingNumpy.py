import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Union, Tuple, Literal

# ImageProcessing class for image manipulation and processing
class ImageProcessing:
    """A class for processing images using NumPy arrays.
    This class provides methods for displaying, cropping, rotating, flipping, converting to grayscale,
    binarizing, convolving, blurring, and sharpening images.
    
    Attributes:
        arr (ndarray): The image data as a NumPy array.
    
    Methods:
        image(): Displays the image stored in the instance.
        display_img(image_array, text='', position='center'): Displays an image with an optional text label.
        compare_img(*args): Compares multiple images side by side.
        crop_img(coordinate_2, coordinate_1=(0,0)): Crops the image according to the provided coordinates.
        rotate_img(num=1): Rotates the image by 90 degrees clockwise or counterclockwise.
        flip_img(plane='h'): Flips the image horizontally or vertically.
        is_rgb(): Checks if the image is in RGB format.
        negative(): Converts the image to its negative.
        grayscale(): Converts the image to grayscale.
        binarise(threshold=128): Converts the image to binary using a specified threshold.
        convolve2d(image_array, kernel): Applies a 2D convolution to an image array using a specified kernel.
        convolve3d(image_array, kernel): Applies a 3D convolution to an image array using a specified kernel.
        blur_img(kernel): Applies a blur to the image using a specified kernel.
        sharpen_img(kernel): Applies a sharpening filter to the image using a specified kernel.
    
    Raises:
        TypeError: If the image_array is not a NumPy ndarray.
        ValueError: If the image_array does not have the correct dimensions (2D or 3D).
    """

    def __init__(self, image_array: ndarray):
        """
        Initializes the ImageProcessing class with an image array.

        Parameters:
            image_array (ndarray): The image data to be processed, expected as a NumPy array.
        
        Raises:
            TypeError: If image_array is not a NumPy ndarray.
            ValueError: If image_array does not have the correct dimensions (2D or 3D).
        """
        if not isinstance(image_array, ndarray):
            raise TypeError("image_array must be a NumPy ndarray")
        if image_array.ndim not in [2, 3]:
            raise ValueError("image_array must be a 2D or 3D array representing an image")
        
        self.arr = image_array
    
    def image(self) -> None:
        """
        Returns the image corresponding to the image array stored in the instance.

        Returns:
            None
        """
        plt.imshow(self.arr, cmap='gray')
        plt.axis('off')
        plt.show()
        return
    
    @staticmethod
    def display_img(image_array: ndarray, text: str = '', position: Literal['center', 'left', 'right'] = 'center') -> None:
        """
        Displays the image with an optional text label.

        Parameters:
            image_array (ndarray): The image data to be displayed, expected as a NumPy array.
            text (str, optional): Text to display with the image. Default is an empty string - no text.
            postion (str, optional): Position of the text. Default is 'center'. 
            Must be one of 'center', 'left', or 'right'.
        
        Returns:
            None

        Raises:
            ValueError: If the image array does not have the correct dimensions (2D or 3D).
            ValueError: If position is not one of 'center', 'left', or 'right'.
            TypeError: If image_array is not a NumPy ndarray.
        """
        if not isinstance(image_array, ndarray):
            raise TypeError("image_array must be a NumPy ndarray")
        if image_array.ndim not in [2, 3]:
            raise ValueError("image_array must be a 2D or 3D array representing an image")
        if image_array.shape == 4:
            # If the image has an alpha channel, remove it
            image_array = image_array[:, :, :3]
        if position not in ['center', 'left', 'right']:
            raise ValueError("position must be one of 'center', 'left', or 'right'")
        
        plt.imshow(image_array, cmap='gray')

        if text != '':
            plt.title(text, fontsize=12, pad=10, loc=position)

        plt.axis('off')
        return

    @staticmethod
    def compare_img(*args: Union[ndarray, Tuple[ndarray, str]]) -> None:
        """
        Compares multiple images side by side.

        Parameters:
            *args (Union[ndarray, Tuple[ndarray, str]]): A variable number of image arrays or tuples containing an image array and a text label.
                If a tuple is provided, the first element should be the image array and the second element should be the text label.
        
        Returns:
            None
        
        Raises:
            TypeError: If any of the arguments are not a NumPy ndarray or a tuple containing a NumPy ndarray and a string.
        """
        if not all(isinstance(arg, (ndarray, tuple)) for arg in args):
            raise TypeError("All arguments must be either a NumPy ndarray or a tuple containing a NumPy ndarray and a string")
        
        num = len(args)
        plt.figure(figsize=(10,5*num))  # Adjust the figure height based on the number of images. Wdth is 10.
        rows = (num+1)//2   # Calculate the number of rows needed for the images

        for i in range(num):
            if isinstance(args[i], tuple):
                if len(args[i]) != 2:
                    raise TypeError("If a tuple is provided, it must contain exactly two elements: (image_array, text)")
                elif not isinstance(args[i][0], ndarray) or not isinstance(args[i][1], str):
                    raise TypeError("If a tuple is provided, the first element must be a NumPy ndarray and the second element must be a string")
                elif args[i][0].ndim not in [2, 3]:
                    raise ValueError("The image_array in the tuple must be a 2D or 3D array representing an image")
                
                # Unpack the tuple into image array and text
                image_arr, text = args[i]

            else:
                if args[i].ndim not in [2, 3]:  #type: ignore
                    raise ValueError("The image_array must be a 2D or 3D array representing an image")
                
                image_arr = args[i]
                text = ''

            plt.subplot(rows,2,i+1)
            ImageProcessing.display_img(image_arr, text)    #type: ignore

        plt.tight_layout()
        plt.show()

        return
    
    def crop_img(self, coordinate_2: Tuple[int, int], coordinate_1: Tuple[int, int] = (0,0)) -> ndarray:
        """
        Crops an image according to the provided coordinates

        Parameters:
            image_array(ndarray): The image array to be cropped
            coordinate_2(Tuple[int, int]): Coordinates of the final point.
            coordinate_1(Tuple[int, int], optional): Coordinates of the initial point. DEFAULT is (0,0)

        Returns:
            ndarray: The cropped image array
        
        Raises:
            TypeError: If coordinate_1 or coordinate_2 are not tuples of 2 integers.
            ValueError: If coordinate_1 or coordinate_2 are out of bounds of the image dimensions.
        """
        if not isinstance(coordinate_1, tuple) or not isinstance(coordinate_2, tuple):
            raise TypeError("coordinate_1 and coordinate_2 must be tuples of 2 integers")
        
        if len(coordinate_1) != 2 or len(coordinate_2) != 2:
            raise TypeError("coordinate_1 and coordinate_2 must be tuples of 2 integers")
        
        if not all(isinstance(coord, int) for coord in coordinate_1 + coordinate_2):
            raise TypeError("coordinate_1 and coordinate_2 must be tuples of 2 integers")
        
        if coordinate_1[0] >= coordinate_2[0] or coordinate_1[1] >= coordinate_2[1]:
            coordinate_1, coordinate_2 = coordinate_2, coordinate_1  # Ensure coordinate_1 is top-left and coordinate_2 is bottom-right
        
        if coordinate_1[0] < 0 or coordinate_1[1] < 0 or coordinate_2[0] > self.arr.shape[1] or coordinate_2[1] > self.arr.shape[0]:
            raise ValueError("Coordinates are out of bounds of the image dimensions")
        
        x1, y1 = coordinate_1
        x2, y2 = coordinate_2

        self.arr = self.arr[y1:y2, x1:x2]

        return self.arr
    
    def rotate_img(self, num: int = 1) -> ndarray:
        """
        Rotates the image by 90 degrees clockwise or counterclockwise.
        
        Parameters:
            num (int, optional): The number of 90-degree rotations to apply. Positive values rotate clockwise, negative values rotate counterclockwise.
            Default is 1 (90 degrees clockwise).
            
        Returns:
            ndarray: The rotated image array.

        Raises:
            ValueError: If num is not an integer.
        """
        if not isinstance(num, int):
            raise ValueError("num must be an integer")
        if num < 0:
            num = -num
        elif num > 4:
            num = num % 4
        elif num == 0:
            return self.arr
        
        # Rotate the image by 90 degrees clockwise
        self.arr = np.rot90(self.arr, k=num)
        return self.arr
    
    def flip_img(self, plane: Literal['h', 'v'] = 'h') -> ndarray:
        """
        Flips the image horizontally or vertically.

        Parameters:
            plane (str, optional): The axis to flip the image along. 'h' for horizontal, 'v' for vertical. Default is 'h'.

        Returns:
            ndarray: The flipped image array.
        """
        if plane == 'v':
            self.arr = np.flipud(self.arr)
        elif plane == 'h':
            self.arr = np.fliplr(self.arr)
        else:
            raise ValueError("axis must be either 'h' (horizontal) or 'v' (vertical)")
        
        return self.arr
    
    def is_rgb(self) -> bool:
        """
        Checks if the image is in RGB format.

        Returns:
            bool: True if the image is RGB (3D array with 3 channels), False otherwise.
        """
        if self.ndim < 3:   #type: ignore
            return False
        else:
            return True
        
    def negative(self) -> ndarray:
        """
        Converts the image to its negative.

        Returns:
            ndarray: The negative image array.
        """
        self.arr = (255 - self.arr).astype(np.uint8)
        return self.arr
    
    def grayscale(self) -> ndarray:
        """
        Converts the image to grayscale using the standard luminance formula.

        Returns:
            ndarray: The grayscale image array.
        """
        if self.arr.ndim == 2:
            # If the image is already grayscale, return it as is
            return self.arr
    
        # Grayscale conversion formula: Y = 0.299*R + 0.587*G + 0.114*B
        gray_convert_arr = np.array((0.299, 0.587, 0.114))
        self.arr = np.dot(self.arr, gray_convert_arr).astype(np.uint8)

        return self.arr
    
    def binarise(self, threshold: int = 128) -> ndarray:
        """
        Converts the image to binary using a specified threshold.

        Parameters:
            threshold (int, optional): The threshold value for binarization. Default is 128. Sholud be between 0 and 255.

        Returns:
            ndarray: The binary image array.
        
        Raises:
            TypeError: If the threshold is not an integer.
            ValueError: If the threshold is not between 0 and 255.
        """
        if not isinstance(threshold, int):
            raise TypeError("threshold must be an integer")
        if threshold < 0 or threshold > 255:
            raise ValueError("threshold must be between 0 and 255")
        if self.arr.ndim == 3:
            # Convert to grayscale first if the image is in RGB format
            self.grayscale()
        
        # Binarization
        self.arr = np.where(self.arr >= threshold, 255, 0).astype(np.uint8)
        return self.arr
    
    @staticmethod
    def convolve2d(image_array: ndarray, kernel: ndarray) -> ndarray:
        """
        Applies a 2D convolution to an image array using a specified kernel.

        Parameters:
            image_array (ndarray): The image data to be convolved, expected as a 2D NumPy array.
            kernel (ndarray): The convolution kernel to apply. Must be a 2D array.
        
        Returns:
            ndarray: The convolved array
        
        Raises:
            ValueError: If the image_array does not have the correct dimension, i.e., 2
            ValueError: If the kernel is not a 2D array.
            TypeError: If the image_array is not a Numpy array.
            TypeError: If the kernel is not a Numpy array.
        """
        if not isinstance(kernel, ndarray):
            raise TypeError("kernel must be a NumPy ndarray")
        if not isinstance(image_array, ndarray):    
            raise TypeError("image_array must be a NumPy ndarray")
        if image_array.ndim != 2:
            raise ValueError("image_array must be a 2D array representing a grayscale image. Use convolve3d instead")
        if kernel.ndim != 2:
            raise ValueError("kernel must be a 2D array representing the convolution kernel")
        
        # Flip the kernel for convolution
        # This is necessary because convolution is defined as a cross-correlation operation
        kernel = np.flipud(np.fliplr(kernel))

        # Get the dimensions of the image and kernel
        rows, columns = image_array.shape
        krows, kcols = kernel.shape

        padrows, padcols = krows//2, kcols//2
        output = np.zeros_like(image_array)
        padded_img = np.pad(image_array, ((padrows, padrows), (padcols, padcols)), 'constant', constant_values=0)

        for row in range(rows):
            for col in range(columns):
                region = padded_img[row:row+krows, col:col+kcols]
                output[row, col] = np.sum(region*kernel)
        return output

    @staticmethod
    def convolve3d(image_array: ndarray, kernel: ndarray) -> ndarray:
        """
        Applies a 3D convolution to an image array using a specified kernel.

        Parameters:
            image_array (ndarray): The image data to be convolved, expected as a NumPy array.
            kernel (ndarray): The convolution kernel to apply. Must be a 2D array.
        
        Returns:
            ndarray: The convolved image array.
        
        Raises:
            ValueError: If the image array does not have the correct dimensions (2D or 3D).
            ValueError: If the kernel is not a 2D array.
            TypeError: If image_array is not a NumPy ndarray.
            TypeError: If kernel is not a NumPy ndarray.
        """
        dimensions = image_array.ndim

        if dimensions > 3:
            raise ValueError("image_array must be a 2D or 3D array representing an image")
        if not isinstance(image_array, ndarray):
            raise TypeError("image_array must be a NumPy ndarray")
        if not isinstance(kernel, ndarray):
            raise TypeError("kernel must be a NumPy ndarray")
        if kernel.ndim != 2:
            raise ValueError("kernel must be a 2D array representing the convolution kernel")
        
        if dimensions == 2:
            return ImageProcessing.convolve2d(image_array, kernel)
        
        else:
            channels = image_array.shape[-1]
            output = np.zeros_like(image_array)

            for c in range(channels):
                channel = image_array[:,:,c]
                output[:,:,c] = ImageProcessing.convolve2d(channel, kernel)

            return output
        
    def blur_img(self, kernel: ndarray = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16) -> ndarray:
        """
        Applies a blur to the image using a specified kernel.

        Parameters:
            kernel (ndarray, optional): The kernel to be used for blurring. Default is a Gaussian kernel.
        
        Returns:
            ndarray: The blurred image array.
        
        Raises:
            TypeError: If the kernel is not a Numpy ndarray.
            ValueError: If the kernel is not a 2D array.
        """
        if not isinstance(kernel, ndarray):
            raise TypeError("kernel must be a NumPy ndarray")
        if kernel.ndim != 2:
            raise ValueError("kernel must be a 2D array representing the convolution kernel")
        
        self.arr = ImageProcessing.convolve3d(self.arr, kernel).astype(np.uint8)
        self.arr = np.clip(self.arr, 0, 255)
        return self.arr
    
    def sharpen_img(self, kernel: ndarray = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])) -> ndarray:
        """
        Applies a sharpening filter to the image using a specified kernel.
    
        Parameters:
            kernel (ndarray, optional): The kernel to be used for sharpening. Default is a sharpening kernel.
        
        Returns:
            ndarray: The sharpened image array.
        
        Raises:
            TypeError: If the kernel is not a Numpy ndarray.
            ValueError: If the kernel is not a 2D array.
        """
        if not isinstance(kernel, ndarray):
            raise TypeError("kernel must be a NumPy ndarray")
        if kernel.ndim != 2:
            raise ValueError("kernel must be a 2D array representing the convolution kernel")
        
        self.arr = ImageProcessing.convolve3d(self.arr, kernel).astype(np.uint8)
        self.arr = np.clip(self.arr, 0, 255)
        return self.arr

    def edge_detection(self, kernel: ndarray) -> ndarray:
        """
        Applies an edge detection filter to the image using a specified kernel.

        Parameters:
            
        """