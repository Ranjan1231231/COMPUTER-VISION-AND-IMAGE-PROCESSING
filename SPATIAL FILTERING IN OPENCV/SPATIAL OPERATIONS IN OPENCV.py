#SPATIAL FILTERING IN PILLOW
import matplotlib.pyplot as plt
import numpy as np
import cv2
def plot_image(image1,image2,title_1="Original",title_2="New image"):
    plt.figure(figsize = (10,10))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image1,cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(image2,cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.show()

#LINEAR FILTERING
image=cv2.imread("lenna.png")
# print(image)
# plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# plt.show()

#getting the number of rows and columns in the image
rows,cols,_=image.shape
## Creates values using a normal distribution with a mean of 0 and standard deviation of 15, the values are converted to unit8 which means the values are between 0 and 255
noise=np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
#adding the noise to the image
noisy_image=image+noise
# plot_image(image,noisy_image,title_1 = "original",title_2 = "image plus noise")


#filtering the noise
#creating a kernel
kernal=np.ones((6,6))/36

#filtering the iamge using the kernel
image_filtered=cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernal)
# plot_image(image_filtered,noisy_image,title_1 = "filtered image",title_2 = "image plus noise")
kernel=np.ones((4,4))/16
image_filtered1=cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernel)
# plot_image(image_filtered1,noisy_image,title_1 = "filtered image",title_2 = "image plus noise")


#GAUSSIAN BLUR
image_filtered2=cv2.GaussianBlur(noisy_image,(5,5),sigmaX=4,sigmaY=4)
# plot_image(image_filtered2,noisy_image,title_1 = "filtered image",title_2 = "image plus noise")
# image_filtered3=cv2.GaussianBlur(noisy_image2,(11,11),sigmaX=10,sigmaY=10)
# plot_image(image_filtered3,noisy_image,title_1 = "filtered image",title_2 = "image plus noise")


#IMAGE SHARPENING
kernel=np.array([[-1,-1,-1],
                 [-1,9,-1],
                [-1,-1,-1]])
sharpened=cv2.filter2D(image,-1,kernel)
# plot_image(sharpened,image,title_1 = "sharpened image",title_2 = "image")


#EDGES
img_gray=cv2.imread("barbara.png",cv2.IMREAD_GRAYSCALE)
# print(img_gray)
# plt.imshow(img_gray,cmap = "gray")
# plt.show()

img_gray=cv2.GaussianBlur(img_gray,(3,3),sigmaX=0.1,sigmaY=0.1)
# plt.imshow(img_gray,cmap = "gray")
# plt.show()

ddepth = cv2.CV_16S
grad_x=cv2.Sobel(src=img_gray,ddepth=ddepth,dx=1,dy=0,ksize=3)
# plt.imshow(grad_x,cmap = "gray")
# plt.show()

grad_y=cv2.Sobel(src=img_gray,ddepth=ddepth,dx=0,dy=1,ksize=3)
# plt.imshow(grad_y,cmap = "gray")
# plt.show()

# We can approximate the gradient by calculating absolute values, and converts the result to 8-bit:
abs_grad_x=cv2.convertScaleAbs(grad_x)
abs_grad_y=cv2.convertScaleAbs(grad_y)

grad=cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
# plt.figure(figsize = (10,10))
# plt.imshow(grad,cmap = "gray")
# plt.show()

#MEDIAN
image=cv2.imread("cameraman.jpeg",cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize = (10,10))
# plt.imshow(image,cmap = "gray")
# plt.show()

#applying the median filter by using the median blur  function

filtered_image=cv2.medianBlur(image,5)
# plt.figure(figsize = (10,10))
# plt.imshow(filtered_image,cmap = "gray")
# plt.show()

#THRESHHOLD FUNCTION PARAMETERS
ret,outs=cv2.threshold(src=image,thresh=0,maxval=255,type=cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
# plt.figure(figsize = (10,10))
# plt.imshow(outs,cmap = "gray")
# plt.show()