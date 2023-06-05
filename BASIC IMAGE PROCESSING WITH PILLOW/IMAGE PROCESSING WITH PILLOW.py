
import os
import matplotlib.pyplot as plt
from PIL import ImageOps# it contains several ready made image processing operations
from PIL import Image
import numpy as np

#for concating the two images
def get_concat_h(im1,im2):
    dst=Image.new('RGB',(im1.width+im2.width,im1.height))
    dst.paste(im1,(0,0))
    dst.paste(im2,(im1.width,0))
    return dst

#LOADING THE IMAGE
my_image1="lenna.png"
# cwd=os.getcwd()
# image_path=os.path.join(cwd,my_image1)
# print(image_path)
image=Image.open(my_image1)# we can also use the full image path insted of my_image1


#PLOTTING THE IMAGE
# image.show()
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.show()
# print(image.size)#no of pixles in width and height
# print(image.mode)#rgb mode in this case
im=image.load()#reads the file content, decodes it, and expands the image into memory.
#We can then check the intensity of the image at the  𝑥-th column and  𝑦-th row:
# x=0
# y=1
# print(im[y,x])
# image.save(("lenna.jpg"))

#GRAYSCALE IMAGE, QUATIZATION AND COLOR CHANNELS
image_grey=ImageOps.grayscale(image)#convert to grey scale image
# image_grey.show()
#QUANTIZATION
# image_grey.quantize(256//2)
# image_grey.show()
#Let’s continue dividing the number of values by two and compare it to the original image.
# for n in range(3,8):
#     plt.figure(figsize=(10,10))
#     plt.imshow(get_concat_h(image_grey,  image_grey.quantize(256//2**n)))
#     plt.title("256 QUATISATION LEVEL LEFT VS {} QUATISATION LEVELS RIGHT".format(256//2**n))
    # plt.show()

#COLOR CAHNNELS
baboon=Image.open('baboon.png')
# baboon.show()
red,green,blue=baboon.split()
# get_concat_h(baboon,red).show()
# get_concat_h(baboon,blue).show()
# get_concat_h(baboon,green).show()


#PIL IMAGES INTO NUMPY IMAGES
# array=np.asarray(image)#asarray turns orignal image to a numy array
# print(array)
array=np.array(image)#it create a new copy of the image and the orignal one will remain unmodeified
# print(array.shape)#(it will output (rows,columns,colors)
# print(array)
# print(array[0,0])
# print(array.min(),array.max())#finding the minimum and maximum intensity of the array


#INDEXING
#plotting the array as the image
# plt.figure(figsize=(10,10))
# plt.imshow(array)
# plt.show()


# we can return the first 256 rows corresponding to the top half of the image:
rows=256
# plt.figure(figsize=(10,10))
# plt.imshow(array[0:rows,:,:])
# plt.show()


#return the first 256 columns corresponding to the first half of the image.
columns=256
# plt.figure(figsize=(10,10))
# plt.imshow(array[:,0:columns,:])
# plt.show()

# reassign an array to another variable
A=array.copy()
# plt.imshow(A)
# plt.show()


#If we do not apply the method copy(), the variable will point to the same location in memory. Consider the array B. If we set all values of array A to zero, as B points to A, the values of B will be zero too:
# B=A
# A[:,:,:]=0
# plt.imshow(B)
# plt.show()


#WORKING WITH DIFFRENT COLOUR CHANNELS
baboon_array=np.array(baboon)
# plt.figure(figsize=(10,10))
# plt.imshow(baboons_array)
# plt.show()
    #plotting he channel intensity values of the red channel
# plt.figure(figsize=(10,10))
# plt.imshow(baboon_array[:,:,0],cmap='gray')
# plt.show()
    # Or we can create a new array and set all but the red color channels to zero. Therefore, when we display the image it appears red:
baboon_red=baboon_array.copy()
# baboon_red[:,:,1]=0
# baboon_red[:,:,2]=0
# plt.figure(figsize=(10,10))
# plt.imshow(baboon_red)
# plt.show()
    #creating same for blue
baboon_blue=baboon_array.copy()
# baboon_blue[:,:,0]=0
# baboon_blue[:,:,1]=0
# plt.figure(figsize=(10,10))
# plt.imshow(baboon_blue)
# plt.show()

#CONVERTING THE LENNA PHOTO TO BLUESCALE
lenna=Image.open('lenna.png')
lenna_array=np.array(lenna)
lenna_blue=lenna_array.copy()
lenna_blue[:,:,0]=0
lenna_blue[:,:,1]=0
plt.figure(figsize=(10,10))
plt.imshow(lenna_blue)
plt.show()