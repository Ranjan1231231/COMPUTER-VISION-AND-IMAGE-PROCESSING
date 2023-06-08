#HISTOGRAM AND INTENSITY TRANSFORMAITION
import cv2
import matplotlib.pyplot as plt
import numpy as np
def plot_image(image_1,image_2,title_1="Original",title_2="NewImage"):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1,2,2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()

def plot_hist(old_image,new_image,title_old="Orignal",title_new="New Image"):
    intensity_values=np.array([x for x in range(256)])
    plt.subplot(1,2,1)
    plt.bar(intensity_values,cv2.calcHist([old_image],[0],None,[256],[0,256])[:,0],width=5)
    plt.title(title_old)
    plt.xlabel('Intensity')
    plt.subplot(1,2,2)
    plt.bar(intensity_values,cv2.calcHist([new_image],[0],None,[256],[0,256])[:,0],width=5)
    plt.title(title_new)
    plt.xlabel("Intensity")
    plt.show()

#HISTOGRAMS
#We use cv.calcHist() to generate the histogram. Here are the parameter values:
# cv2.calcHist(CV array:[image] this is the image channel:[0],for this course it will always be [None],the number of bins:[L],the range of index of bins:[0,L-1])

#TOY EXAMPLE
toy_image=np.array([[0,2,2],[1,1,1],[1,1,2]],dtype=np.uint8)
# plt.imshow(toy_image,cmap="gray")
# plt.show()
# print("Toy_image:",toy_image)
# We can use the caclHist function, in this case, we use only three bins as there are only three values, and the index of the bins are from 1 to 3.

# plt.bar([x for x in range(6)],[1,5,2,0,0,0])
# plt.show()

# plt.bar([x for x in range(6)],[0,1,0,5,0,2])
# plt.show()

#GRAY SCALE HISTOGRAMS
goldhill=cv2.imread("goldhill.bmp",cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(10,10))
# plt.imshow(goldhill,cmap="gray")
# plt.show()

hist=cv2.calcHist([goldhill],[0],None,[256],[0,256])

# We can plot it as a bar graph, the  𝑥
#  -axis are the pixel intensities and the  𝑦
#  -axis is the number of times of occurrences that the corresponding pixel intensity value on  𝑥
#  -axis occurred.
intensity_values=np.array([x for x in range(hist.shape[0])])
# plt.bar(intensity_values,hist[:,0],width=5)
# plt.title("Bar histogram")
# plt.show()

#CONVERTING IT TO A PROBAILITY MASS FUNCTION
PMF=hist/(goldhill.shape[0]*goldhill.shape[1])
# plt.plot(intensity_values,hist)
# plt.title("histogram")
# plt.show()

#We can also apply a histogram to each image color channel
baboon=cv2.imread("baboon.png")
# plt.imshow(cv2.cvtColor(baboon,cv2.COLOR_BGR2RGB))
# plt.show()

# In the loop, the value for i specifies what color channel calcHist is going to calculate the histogram for.

color=('blue','green','red')
for i,col in enumerate(color):
    histr=cv2.calcHist([baboon],[i],None,[256],[0,256])
#     plt.plot(intensity_values,histr,color=col,label=col+"channel")
#     plt.xlim([0,256])
# plt.legend()
# plt.title("Histogram Channels")
# plt.show()
#

#INTENSITY TRANSFORMATIONS
#IMAGE NEGETIVES
neg_toy_image=-1*toy_image+255
# print("toy image\n",toy_image)
# print("image negetive\n",neg_toy_image)

# plt.figure(figsize=(10,10))
# plt.subplot(1,2,1)
# plt.imshow(toy_image,cmap="gray")
# plt.subplot(1,2,2)
# plt.imshow(neg_toy_image,cmap="gray")
# plt.show()
# print("toy_image:",toy_image)

#Reversing image intensity has many applications, including making it simpler to analyze medical images. Consider the mammogram with micro-calcifications on the upper quadrant:

image=cv2.imread('mammogram.png',cv2.IMREAD_GRAYSCALE)
cv2.rectangle(image,pt1=(160,212),pt2=(250,289),color=(255),thickness=2)
# plt.figure(figsize=(10,10))
# plt.imshow(image,cmap="gray")
# plt.show()

#applying the intensity transformation
img_neg=-1*image+255
# plt.figure(figsize=(10,10))
# plt.imshow(img_neg,cmap="gray")
# plt.show()

#BRIGHTNESS AND CONTRAST ADJUSTMENTS
alpha=1#Simple cntrast control
beta=100#simple brightness control
new_image=cv2.convertScaleAbs(goldhill,alpha=alpha,beta=beta)

#PLOTTING THE BRIGHTER IMAGE
# plot_image(goldhill,new_image,title_1="Orignla",title_2="brightness control")

#BRIGHTER IMAGE HISTOGRAM
# plt.figure(figsize=(10,5))
# plot_hist(goldhill,new_image,"orignal","brightness control")

#INCREASING THE CONTRAST BY ALPHA
# plt.figure(figsize=(10,50))
alpha=2#simple contrast control
beta=0#simple brightness control
new_image1=cv2.convertScaleAbs(goldhill,alpha=alpha,beta=beta)
# plot_image(goldhill,new_image1,"Orignal","Contrast control")
# plt.figure(figsize=(10,5))
# plot_hist(goldhill,new_image1,"Orignal","contrast control")

#making the image darker and increasing the contrast at the sae time

alpha=3 #simple contrast control
beta=-200 #simple brightness control
new_image2=cv2.convertScaleAbs(goldhill,alpha=alpha,beta=beta)
# plt.figure(figsize=(10,5))
# plot_image(goldhill,new_image2,"Original","Brightnerss and contrast control")
# plt.figure(figsize=(10,5))
# plot_hist(goldhill,new_image2,"orignial","brightness and contrast control")


#HISTOGRAM EQUALIZATION #

### IMPORTANT###

zelda=cv2.imread("zelda.png",cv2.IMREAD_GRAYSCALE)
# new_image3=cv2.equalizeHist(zelda)
# plot_image(zelda,new_image3,"orignal","Histogram Eualization")
# plt.figure(figsize=(10,5))
# plot_hist(zelda,new_image3,"orignal","histogram equalization")

#THRESHHOLDING AND SIMPLE SEGMENTATION
def thresholding(input_img,threshold,max_value=255,min_value=0):
    N,M=input_img.shape
    image_out=np.zeros(((N,M)),dtype=np.uint8)
    for i in range(N):
        for j in range(M):
            if input_img[i,j]>threshold:
                image_out[i,j]=max_value
            else:
                image_out[i,j]=min_value
    return image_out
#We can apply thresholding, by setting all the values less than two to zero.
threshold=1
max_value=2
min_Value=0
thresholding_toy=thresholding(toy_image,threshold=threshold,max_value=max_value,min_value=min_Value)

#compairing the two images
# plt.figure(figsize=(10,10))
# plt.subplot(1,2,1)
# plt.imshow(toy_image,cmap="gray")
# plt.title("Orignal Image")
# plt.subplot(1,2,2)
# plt.imshow(thresholding_toy,cmap="gray")
# plt.title("image after thresholding")
# plt.show()

#USING THE CAMERAMAN IMAGE
image=cv2.imread("cameraman.jpeg",cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(10,10))
# plt.imshow(image,cmap="gray")
# plt.show()


#WE CAN SEE THE TWO HISTOGRAM AS TWO PEEKS , THIS MEANS THAT THERE IS A LARGE PROPORTION OF PIXZELS IN THOSE TWO RANGES
hist=cv2.calcHist([goldhill],[0],None,[256],[0,256])
# plt.bar(intensity_values,hist[:,0],width=5)
# plt.title("Bar histogram")
# plt.show()

## The cameraman corresponds to the darker pixels, therefore we can set the Threshold in such a way as to segment the cameraman. In this case, it looks to be slightly less than 90, let’s give it a try:


threshold=87
max_value=225
min_Value=0
new_image5=thresholding(image,threshold=threshold,max_value=max_value,min_value=min_Value)
# plot_image(image,new_image5,"orignal","image after threshholding")
# plt.figure(figsize=(10,5))
# plot_hist(image,new_image5,"Orignal","IMAGE AFTER THRESHHOLDING")
#



##The function cv.threshold Applies a threshold to the gray image, with the following parameters:
# cv.threshold(grayscale image, threshold value, maximum value to use, thresholding type )
# The parameter thresholding type is the type of thresholding we would like to perform. For example, we have basic thresholding: cv2.THRESH_BINARY this is the type we implemented in the function thresholding, it just a number:
ret,new_image6=cv2.threshold(image,threshold,max_value,cv2.THRESH_BINARY)
# plot_image(image,new_image6,"original","Image after thresholding")
# plot_hist(image,new_image,"Orignal","Image after thresholding")

# ret is the threshold value and new_image is the image after thresholding has been applied. There are different threshold types, for example, cv2.THRESH_TRUNC will not change the values if the pixels are less than the threshold value:

ret,new_image7=cv2.threshold(image,86,255,cv2.THRESH_TRUNC)
# plot_image(image,new_image7,"Orignal","Image after thresholding")
# plot_hist(image,new_image7,"orignal","image afterthresholding")

# Otsu's method cv2.THRESH_OTSU avoids having to choose a value and determines it automatically, using the histogram.
ret,otsu=cv2.threshold(image,0,255,cv2.THRESH_OTSU)
plot_image(image,otsu,"Original","Otsu's method")
plot_hist(image,otsu,"original","otsu's method")