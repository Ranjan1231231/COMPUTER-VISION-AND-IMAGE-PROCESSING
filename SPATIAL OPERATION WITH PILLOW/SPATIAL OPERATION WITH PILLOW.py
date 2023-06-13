#SPATIAL FILTERING IN PILLOW
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFilter
from PIL import Image
def plot_image(image1,image2,title_1="Original",title_2="New image"):
    plt.figure(figsize = (10,10))
    plt.subplot(1,2,1)
    plt.imshow(image1)
    plt.title(title_1)
    plt.subplot(1,2,2)
    plt.imshow(image2)
    plt.title(title_2)
    plt.show()

#LINEAR FILTERING
image=Image.open("lenna.png")
# plt.figure(figsize = (5,5))
# plt.imshow(image)
# plt.show()

# The images we are working with are comprised of RGB values, which are values from 0 to 255. Zero means white noise, this makes the image look grainy:
#get the no of rows and cols in the image
rows,cols=image.size
# # Creates values using a normal distribution with a mean of 0 and standard deviation of 15, the values are converted to unit8 which means the values are between 0 and 255
noise=np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
#add the noise to the image
noisy_image=image+noise
#creates a pil image from an array
noisy_image1=Image.fromarray(noisy_image)
#plots the original image and the image with noise using the function defined at the top
# plot_image(image,noisy_image1,title_1 = "original",title_2 = "image plus noise")


#FILTERING NOISE

#creating a kernel which is a 5 by 5 array where each value is 1/36
kernel=np.ones((5,5))/36
#creating a imagefilterkernal by providing the kernel size and flattend kernel
kernel_filter=ImageFilter.Kernel((5,5),kernel.flatten())
#filtering the image using the kernel
image_filtered=noisy_image1.filter(kernel_filter)
# plot_image(image_filtered,noisy_image1,title_1 = "FILTERED IMAGE",title_2 = "IMAGE PLUS NOISE")

#creating a kernel which is a 3/3 array where each value is 1/36
kernel=np.ones((3,3))/36
#creating an image filter
kernel_filter=ImageFilter.Kernel((3,3),kernel.flatten())
#filter te image using the kernel
image_filtered1=noisy_image1.filter(kernel_filter)
# plot_image(image_filtered1,noisy_image1,title_1 = "filtered image",title_2 = "image plus noise")

#GAUSSIAN BLUR
#FILTERING THE IMAGE USING GAUSSIAN BLUR
image_filtered2=noisy_image1.filter(ImageFilter.GaussianBlur)
# plot_image(image_filtered2,noisy_image1,title_1 = "filtered image",title_2 = "image plus noise")

#IMAGE SHARPNING
#MAKING A COMMPN KERNEL FOR IMAGE SHARPNING
kernel=np.array([[-1,-1,-1],
                  [-1,9,-1],
                  [-1,-1,-1]])
kernel=ImageFilter.Kernel((3,3),kernel.flatten())
#applying the sharpening filter using kernel on the original image without noise
sharpned=image.filter(kernel)
# plot_image(sharpned,image,title_1 = "sharpned image",title_2 = "orignal image")


#sharping using predefined filters from PIL
sharpened=image.filter(ImageFilter.SHARPEN)
# plot_image(sharpened,image,title_1 = "SHARPENED",title_2 = "ORIGNAL IMAGE")

#EDGES
img_gray=Image.open("barbara.png")
# plt.imshow(img_gray,cmap = "gray")
# plt.show()
#filtering the image using EDGE_ENHANCE filter
img_gray=img_gray.filter(ImageFilter.EDGE_ENHANCE)
# plt.imshow(img_gray,cmap="gray")
# plt.show()

#filtering the edges using find edges filter
img_gray=img_gray.filter(ImageFilter.FIND_EDGES)
# plt.figure(figsize = (10,10))
# plt.imshow(img_gray,cmap = "gray")
# plt.show()


#MEDIAN

image=Image.open("cameraman.jpeg")
# plt.figure(figsize = (10,10))
# plt.imshow(image,cmap = "gray")
# plt.show()
image=image.filter(ImageFilter.MedianFilter)
plt.figure(figsize = (10,10))
plt.imshow(image,cmap = "gray")
plt.show()