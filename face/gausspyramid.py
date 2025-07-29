import os
import numpy as np
import cv2

# read image from train folder
def readImage(imageName):
    image_path = os.path.join("/Users/sareenamann/AETHER/face/", imageName)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file '{image_path}' does not exist")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image '{image_path}'")
    return image

# create folder to store results in
def createFolder(name):
    print(f"Creating folder: {name}")
    if not os.path.exists(name):
        os.makedirs(name)
        print(f"Created folder with the name \"{name}\"")
    else:
        print(f"Folder \"{name}\" already exists!")

def createResultFolders(imageName):
    name = "./results/"+imageName.split(".")[0]+"_results"
    createFolder(name)
    createFolder(name+"/gaussian_pyramid")
    # createFolder(name+"/laplacian_pyramid")
    # createFolder(name+"/log_pyramid")
    return name

# convert rgb to greyscale (if needed)
def rgb2gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def convert2GrayScaleIfNeeded(image):
    if(len(image.shape)>2):
        print("Converting to Greyscale..")
        return rgb2gray(image)
    else:
        print("Returning already Grayscale image!")
        return image

# convolution function (for gaussian pyramid and laplace pyramid, used in following functions with respective array)
def convolve(image, imageFilter, scaleValue):
    blurredList = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filterPixels = imageFilter[max(0, 1-i):imageFilter.shape[0]+min(0, image.shape[0]-(i+2)), max(0, 1-j):imageFilter.shape[1]+min(0, (image.shape[1]-(j+2)))]
            startX = ((j+max(0, 1-j))-1)
            startY = ((i+max(0, 1-i))-1)
            imagePixels = image[startY:startY+filterPixels.shape[0], startX:startX+filterPixels.shape[1]]
            flatImage = np.array(filterPixels).flatten()
            flatFilter = np.array(imagePixels).flatten()
            blurredPixel = (np.dot(flatImage, flatFilter))/scaleValue
            blurredList.append(blurredPixel)
    blurredImage = np.array(blurredList).reshape(image.shape)
    return blurredImage

gaussianKernel = np.array([[1, 2, 1],[ 2, 4, 2], [1, 2, 1]])
gaussianScale = 16.0

def gaussianBlur(image):
    return convolve(image, gaussianKernel, gaussianScale)

# changes the size of image (to reduce pixels, smaller blurred with fewer pixels and larger high resolution with more pixels images)
def scaleDownImage(image):
    return image[1:image.shape[0]:2, 1:image.shape[1]:2]

def scaleUp(image):
    return np.insert(np.insert(image, np.arange(1, image.shape[0]+1), 0, axis=0), np.arange(1, 
                image.shape[1]+1), 0, axis=1)

def scaleUpImage(image):
    scaledImage = scaleUp(image)
    scaledUpImage = gaussianBlur(scaledImage)*4
    return scaledUpImage

# returns the difference of two images - used to get the laplacian
def imageDifference(image_1, image_2):
    return np.subtract(image_1[0:image_2.shape[0], :image_2.shape[1]], image_2)

# constructs the gaussian, laplacian, and log pyramids
def constructPyramids(image, imageLabel, folderName, N=5):
    levelImage = image
    gaussianPath = folderName+"/gaussian_pyramid/"+imageLabel+"_gaussian_level_"

    pyramid = []
    for i in range(N):
        blurredLevelImage = gaussianBlur(levelImage)
        scaledDownLevelImage = scaleDownImage(blurredLevelImage)
        pyramid.append(scaledDownLevelImage)
        cv2.imwrite(gaussianPath + str(i) + ".jpg", scaledDownLevelImage)
        scaledUpLevelImage = scaleUpImage(scaledDownLevelImage)
        differenceImage = imageDifference(levelImage, scaledUpLevelImage)
        levelImage = scaledDownLevelImage
    return pyramid