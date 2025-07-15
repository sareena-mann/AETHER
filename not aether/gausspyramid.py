import os
import numpy as np
import cv2

# read image from train folder
def readImage(imageName):
    return cv2.imread("./not aether/"+imageName)

# create folder to store results in
def createFolder(name):
    if not os.path.exists(name):
        os.makedirs(name)
        print(f"Created folder with the name \"{name}\"")
    else:
        print(f"Folder \"{name}\" already exists!")
#def createFolder(name):
    #try:
        #os.mkdir(name)
        #print("Created folder with the name \"{}\"".format(name))
        #return name
    #except FileExistsError:
        #print("Folder \"{}\" already exists!".format(name))

def createResultFolders(imageName):
    name = "./results/"+imageName.split(".")[0]+"_results"
    createFolder(name)
    createFolder(name+"/gaussian_pyramid")
    createFolder(name+"/laplacian_pyramid")
    createFolder(name+"/log_pyramid")
    return name

# image to test/run
# imageName = "your_image_name"
imageName="image1.jpg"
image = readImage(imageName)
imageLabel = imageName.split(".")[0]
folderName = createResultFolders(imageName)

# convert rgb to greyscale (if needed)
def rgb2gray(image):
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

greyImage = convert2GrayScaleIfNeeded(image)

# array to convolve image with for gaussian pyramid
gaussianKernel = np.array([[1, 2, 1],[ 2, 4, 2], [1, 2, 1]])
gaussianScale = 16.0

# array to convolve image with for laplace pyramid
laplacianKernel = np.array([[0, 1, 0],[ 1, -4, 1], [0, 1, 0]])
laplacianScale = 1.0

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

def gaussianBlur(image):
    return convolve(image, gaussianKernel, gaussianScale)

def laplacian(image):
    return convolve(image, laplacianKernel, laplacianScale)

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
    laplacianPath = folderName+"/laplacian_pyramid/"+imageLabel+"_laplacian_level_"
    logPath = folderName+"/log_pyramid/"+imageLabel+"_log_level_"
    for i in range(N):
        blurredLevelImage = gaussianBlur(levelImage)
        scaledDownLevelImage = scaleDownImage(blurredLevelImage)
        cv2.imwrite(gaussianPath+str(i)+".jpg", scaledDownLevelImage)
        scaledUpLevelImage = scaleUpImage(scaledDownLevelImage)
        differenceImage = imageDifference(levelImage, scaledUpLevelImage)
        cv2.imwrite(laplacianPath+str(i)+".jpg", differenceImage)
        logLevelImage = laplacian(blurredLevelImage)
        cv2.imwrite(logPath+str(i)+".jpg", logLevelImage)
        levelImage = scaledDownLevelImage

constructPyramids(greyImage, imageLabel, folderName)