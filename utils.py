
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
from PIL import Image


############################################################
############### Data Loading Utils Functions ###############
############################################################

def convert_png_to_array(img):
    '''
    Input:
        img <- a path to a .png image
    Output:
        array_img <- a numpy matrix (64, 64)
    '''
    
    png = Image.open(img)
    array_img = np.asarray(png)

    return array_img

def get_img(data_path, img_size, greyscale):
    # Getting image array from path:
    img = imread(data_path)
    img = resize(img, (img_size, img_size, 1 if greyscale else 3))
    return img


def get_dataset():
    '''
    This functions loads the data from file and returns:
        
        X: (2062, 4096) matrix with test images as flattened numpy arrays with pixels range [0, 1]
        
        X_unf: (2062, 64, 64) matrix with test images as (64, 64) numpy matrix
        
        X_e: (2062, 4096) matrix with test images converted to PNG and processed with edge detection and 
             reconverted to flattened numpy arrays with pixels range [0, 255]
             
        Y: matrix of the true labels
             
    '''
  
    X = np.load('X.npy')
    Y = np.load('Y.npy')
   
    X_unf = np.copy(X)    
    
    X = np.array([X[i].flatten() for i in range(len(X))])
    
    Y = np.where(Y==1)[1]
    permutation = {0:9, 1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}
    for i in range(len(Y)):
        Y[i] = permutation[Y[i]]
        
    X_e = [] 
    for i in range(len(X_unf)):
    
        path = f'./png/train{i}.png'
        X_e.append(convert_png_to_array(path))

    X_e = np.array([X_e[i].flatten() for i in range(len(X_e))])
        
    return X, X_unf, Y, X_e

##########################################################
############### Image Processing Functions ###############
##########################################################

def cleaning(img, rows):
    
    for row in rows:
        img[row][0:10] = 1
        img[row][53:64] = 1
        
    return img

def image_to_bw(image, threshold = 0.5):
    '''
    Convert an image to black and white by comparing every pixel to a threshold

    Params:
        [image] : image to process
        [threshold] : threshold that will define each pixel output. 
            Pixels with a greater value than threshold will be black, white otherwise.
    '''
    new_img = []
    for el in image.flat: 
        if el > threshold: new_img.append(1)
        else: new_img.append(0)
        
    return np.array(new_img)