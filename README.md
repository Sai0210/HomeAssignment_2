## Home_Assignment_2
## Student Name : Sai krishna Edara
## Student Id : 700769262
## Question 2

### Convolution Example using TensorFlow and Keras

This project demonstrates how to perform 2D convolutions on an input matrix using TensorFlow and Keras.

### Requirements

- Python 3.x
- NumPy
- TensorFlow

### Code breakdown
 Define Input Matrix and Kernel using np.array()

### Reshape Input and Kernel
The input_matrix and kernel are reshaped using input_matrix.reshape() and kernel.reshape()

### Perform Convolution function
Here we define and perform the Convolution function using the input_data and the kernel_data for different strides and padding

### Perform Convolutions with different parameters
Here we give the defined convolution function different strides and padding

### Expected Output

The script prints

Convolution Results when Stride = 1 and padding is 'VALID'

Convolution Results when Stride = 1 and padding is 'SAME'

Convolution Results when Stride = 2 and padding is 'VALID'

Convolution Results when Stride = 2 and padding is 'SAME'


## Question 3 
### Sobel Edge Detection and Pooling Operations
This project demonstrates how to perform Sobel edge detection on an image and apply max and average pooling operations on a random matrix using OpenCV, TensorFlow, and other libraries

### Overview
This project demonstrates image processing techniques, including edge detection using Sobel filters and pooling operations (Max Pooling and Average Pooling). The script performs the following tasks:

Downloads an image from a given URL.
Converts the image to grayscale.
Applies Sobel filters to detect edges in the x and y directions.
Visualizes the original and processed images.
Demonstrates Max Pooling and Average Pooling using a randomly generated 4×4 matrix.

### Requirements
To run this script, you need the following dependencies:

Python 3.x
OpenCV (cv2)
TensorFlow
NumPy
Matplotlib
Requests
Pillow (PIL)

### Usage
Clone this repository or copy the script.
Replace the image_url variable with your desired image URL.

### 1.How it works
#### Image Processing with Sobel Filters
The script loads an image from a URL and converts it to grayscale.

Sobel edge detection is applied using the following kernels:

Sobel X Kernel:

[[-1,  0,  1]  
 [-2,  0,  2]  
 [-1,  0,  1]]

 Sobel Y Kernel:
 
 [[-1, -2, -1],  
 [ 0,  0,  0],  
 [ 1,  2,  1]]

 The processed images are displayed using matplotlib.

 ### 2.Pooling Operations
A 4×4 random matrix is generated.
Max Pooling and Average Pooling are applied using TensorFlow's MaxPooling2D and AveragePooling2D layers with a 2×2 pool size and stride (2,2).
The resulting 2×2 pooled matrices are displayed in the console.

### Example Output
#### Sobel Edge Detection Visualization
The script displays:

The original grayscale image

The Sobel X edge detection result

The Sobel Y edge detection result

### Pooling Output Example
Original Matrix:

 [[0.21  0.45  0.76  0.89]  
  [0.34  0.67  0.23  0.54]  
  [0.90  0.32  0.68  0.77]  
  [0.11  0.88  0.94  0.15]]

Max Pooled Matrix:

 [[0.67  0.89]  
  [0.90  0.94]]

Average Pooled Matrix:

 [[0.4175 0.605 ]  
  [0.5525 0.635 ]]

  ## Question 4
  ### Deep Learning Models: AlexNet and ResNet Implementation
  #### Overview
  This project implements two popular deep learning architectures using TensorFlow and Keras:

  ##### AlexNet:
A convolutional neural network (CNN) designed for image classification.
  ##### ResNet: 
A deep residual network that uses residual connections to overcome vanishing gradient problems.

#### Requirements
Ensure you have the following dependencies installed:

TensorFlow
NumPy

#### Usage
Run the script to create and display the model architectures:
python script.py

### 1. AlexNet Implementation
#### Architecture Details
AlexNet is a deep convolutional network designed for large-scale image classification. It consists of:

5 Convolutional Layers with ReLU activation.
Max Pooling Layers for dimensionality reduction.
2 Fully Connected Layers with 4096 neurons each.
Dropout Layers to reduce overfitting.
Softmax Output Layer for 10-class classification.
#### Code Breakdown
##### First Convolutional Layer:
Uses a 96-filter (11x11) kernel with a stride of 4 to extract features from 227x227x3 input images.
##### Max Pooling:
A 3x3 filter with a stride of 2 reduces spatial dimensions.
##### Second Convolutional Layer:
Uses 256 filters with a 5x5 kernel and same padding to maintain spatial dimensions.
Another Max Pooling Layer reduces size.
##### Three Consecutive Convolutional Layers:
Each using 3x3 kernels, the first two with 384 filters and the last with 256 filters.
Final Max Pooling Layer reduces dimensions before flattening.
##### Fully Connected Layers:
Two layers with 4096 neurons, followed by dropout to prevent overfitting.
##### Output Layer:
A softmax layer with 10 classes for classification.
#### Model Summary
After running the model summary, the layer-wise structure of AlexNet is displayed, showing the number of parameters and output shapes.

### 2. ResNet Implementation
#### Architecture Details
ResNet (Residual Network) introduces skip connections (residual connections) to solve the vanishing gradient problem in deep networks. This implementation includes:

Residual Blocks: Skip connections to allow gradient flow.
Convolutional and Max Pooling Layers for feature extraction.
Fully Connected Layers for classification.

#### Code Breakdown
##### Input Layer:
Takes in 224x224x3 images.
##### Initial Convolutional Layer:
A 64-filter (7x7) kernel with a stride of 2, followed by Max Pooling.
##### Residual Block:
First convolutional layer with 3x3 kernel and ReLU activation.
Second convolutional layer with another 3x3 kernel.
Skip connection that adds the input directly to the output before applying activation.
##### Second Residual Block:
Another 64-filter block with the same structure.
##### Flatten Layer:
Converts the tensor into a vector.
##### Fully Connected Layer:
A 128-neuron dense layer.
##### Output Layer:
A softmax layer with 10 classes for classification.

#### Model Summary
After running the model summary, the layer-wise structure of ResNet is displayed, showing the number of parameters and output shapes.
