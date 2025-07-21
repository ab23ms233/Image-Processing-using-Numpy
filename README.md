# 🖼️ Image Processing Using NumPy

This project demonstrates how to perform various image processing operations using only NumPy and basic Python libraries — without relying on OpenCV or other high-level image libraries. It's designed to help you understand the mechanics of image transformations through direct pixel-level manipulation.

---

# 🧠 How This Project Works

This project is built around the idea that an image is nothing but a grid of pixel values — and NumPy arrays are the perfect structure to work with such data. Here's how the key operations in this project function under the hood:

---

## 🖼️ Images as NumPy Arrays

1. **Pixel Grid Representation**: An image can be represented as a NumPy array where each cell corresponds to a pixel. The values in the array define the **intensity** or **color** of each pixel.

2. **Grayscale Images**: Each pixel holds a single integer between **0 and 255**, where:
   - `0` = black
   - `255` = white
   - values in between = shades of gray

3. **Image Dimensions**: A `100x100` pixel image will be a **2D array** with 100 rows and 100 columns, each representing one pixel.

---

## 🌈 RGB Images and Channels

4. **RGB Format**: A color image is stored as a **3D array**:  
   - Dimension 1: height (rows)  
   - Dimension 2: width (columns)  
   - Dimension 3: color channels (Red, Green, Blue)

Each pixel is represented by a triplet of integers `[R, G, B]`, each in the 0–255 range.

---

## 🔄 Basic Operations

5. **Rotate**: Rotating the image is equivalent to rotating the 2D array (or each channel in a 3D array) by 90° clockwise or counter-clockwise.

6. **Flip**: Flipping an image horizontally or vertically corresponds to flipping the array along an axis using NumPy functions like `np.flipud` or `np.fliplr`.

7. **Crop**: Cropping means slicing a subregion from the original array using standard NumPy indexing.

---

## 🎨 Color Transformations

8. **Negative**: Creating a negative image involves subtracting each pixel value from 255:  
   ```
   new_pixel = 255 - original_pixel
   ```

9. **Grayscale**: Converting a color image to grayscale compresses the 3D array into 2D by computing a weighted sum of RGB values using the **luminescence formula**:  
   ```
   Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
   ```  
   This accounts for how the human eye perceives brightness from each color.

10. **Binarisation**: All pixels below a threshold (e.g., 128) are set to `0`, and those above are set to `255`. This turns the image into black and white (useful for edge detection, masks, etc.).

---

## 🧮 Convolution & Filters

11. **What is Convolution?**  
    Convolution is a technique where a **small matrix (kernel)** is moved across the image, and for each location, a weighted sum of the neighboring pixels is computed.  
    This allows us to apply effects like blur, sharpen, or detect edges.

    You can read more about convolution at: https://samruddhichitnis02.medium.com/how-to-do-image-convolution-from-scratch-using-numpy-902541f08a9a

---

## ✨ Image Effects Using Kernels

12. **Sharpening**:  
    A sharpening kernel highlights the central pixel and subtracts its surroundings. Example:
    ```
    [[ 0, -1,  0],
     [-1,  5, -1],
     [ 0, -1,  0]]
    ```
    This boosts contrast between a pixel and its neighbors — making edges and details stand out.

13. **Blurring (Gaussian Blur)**:  
    Blurring smoothens the image by averaging surrounding pixels. A **Gaussian kernel** gives more weight to the center and less to the edges:
    ```
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]] / 16
    ```

14. **Edge Detection**:  
    Uses **Sobel filters** (`sobel_x` and `sobel_y`) to detect intensity changes in horizontal and vertical directions:
    - `sobel_x` highlights vertical edges
    - `sobel_y` highlights horizontal edges  

    The final edge map is computed using:
    ```
    Edges = sqrt(Gx^2 + Gy^2)
    ```
    where `Gx` and `Gy` are results of convolutions with `sobel_x` and `sobel_y` respectively.

-----

## 📂 Repository Structure

| File/Folder              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `Class_ImgProcessing.py` | Contains the `ImageProcessing` class that defines all image operations.     |
| `Program_ImgProcessing.py` | Interactive terminal-based image processing application using the class.  |
| `ImageProcessing_Demo.ipynb` | Jupyter notebook to demonstrate each function with examples.         |
| `dog.jpg`, `swan.jpg`    | Sample images used for processing and demonstrations.                      |
| `environment.yaml`       | Conda environment file to recreate the development environment.            |
| `requirements.txt`       | List of Python packages used in this project (pip format).                 |

---

## ✨ Features

The `ImageProcessing` class supports the following operations:

- 📷 Display image, compare images side-by-side
- ✂️ Crop
- 🔄 Rotate (clockwise & counter-clockwise)
- ↕️ Flip (horizontal & vertical)
- 🎨 Convert to Grayscale or Negative
- 🔲 Binarisation with thresholding
- 🧹 Blur (Gaussian kernel)
- 🔪 Sharpen
- ⚡ Edge Detection (Sobel filters)
- 🎛️ Custom 2D and 3D convolution
- 🧪 Generate synthetic striped images with blending

---

## 🧰 Technologies Used

- **NumPy** – for all core image processing operations, including slicing, thresholding, and mathematical transformations.
- **PIL (Pillow)** – for loading and converting images to NumPy arrays.
- **Matplotlib** – for displaying images and visual comparisons.
- **SciPy (`scipy.ndimage.convolve`)** – for faster and flexible convolution on 2D and 3D arrays.
- **Colorama** - for an interactive terminal with coloured texts

---

## 🚀 Getting Started

### 🔧 1. Clone the repository

```bash
git clone https://github.com/your-username/Image-Processing-Using-Numpy.git
cd Image-Processing-Using-Numpy
```

### 🐍 2. Create and activate the Conda environment

```bash
conda env create -f environment.yaml
conda activate image_processing
```

*Alternatively*, if using `pip`:

```bash
pip install -r requirements.txt
```

### ▶️ 3. Run the image processing program

```bash
python Program_ImgProcessing.py
```

### 📓 4. Or explore the notebook interactively

```bash
jupyter notebook ImageProcessing_Demo.ipynb
```

---

## 📸 Preview

<img src="swan.jpg" width="300"> <img src="dog.jpg" width="300">

---

## 📚 License

This project is open-source and free to use.

---

## 🙋‍♂️ Author

**Arya Basak**  
Student | Python & ML Enthusiast  
Feel free to contribute, suggest improvements, or raise issues!
