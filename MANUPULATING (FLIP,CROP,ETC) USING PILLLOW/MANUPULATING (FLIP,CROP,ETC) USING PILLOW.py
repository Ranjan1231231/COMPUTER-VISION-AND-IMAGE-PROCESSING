from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageFont

# def get_concat_h(im1, im2):
#     dst = Image.new('RGB', (im1.width + im2.width, im1.height))
#     dst.paste(im1, (0, 0))
#     dst.paste(im2, (im1.width, 0))
#     return dst

#ASSIGNING MEMORY POINTER VS COPYING
baboon=np.array(Image.open("baboon.png"))
# plt.figure(figsize=(5,5))
# plt.imshow(baboon)
# plt.show()

#assigning the pointer
A=baboon
# print(id(A)==id(baboon))#check the memory location of A AND BABOON IS SAME OR NOT

#copying
B=baboon.copy()
# print(id(B)==id(baboon))
baboon[:,:,] = 0
# plt.figure(figsize=(10,10))
# plt.subplot(121)
# plt.imshow(baboon)
# plt.title("baboon")
# plt.subplot(122)
# plt.imshow(A)
# plt.title("array A")
# plt.show()
# plt.figure(figsize=(10,10))
# plt.subplot(121)
# plt.imshow(baboon)
# plt.title("baboon")
# plt.subplot(122)
# plt.imshow(B)
# plt.title("array B")
# plt.show()


#FLIPPING IMAGES
image=Image.open("cat.png")
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.show()

array=np.array(image)
width,height,C=array.shape
# print('width = {},height = {},c = {}'.format(width,height,C))

#TRADITIONAL APPROACH TO FLIP THE IMAGE BY ASSIGNING THE SAME SIZE OF THE ORIGINAL IMAGE TO A SECOND ARRAY
array_flip=np.zeros((width,height,C),dtype=np.uint8)
# print(array_flip)
for i,row in enumerate(array):
    array_flip[width-1-i,:,:]=row
    # print(array_flip)

#FLIPPING THE IMAGE
im_flip=ImageOps.flip(image)
# plt.figure(figsize=(10,10))
# plt.imshow(im_flip)
# plt.show()

#MIRRORING THE IMAGE
im_mirror=ImageOps.mirror(image)
# plt.figure(figsize=(10,10))
# plt.imshow(im_mirror)
# plt.show()

#FLIPPING THE IMAGE THROUGH TRANSPOSE METOD FROM THE ARRAY
im_flip=image.transpose(4)
# plt.imshow(im_flip)
# plt.show()

flip={
    "FLIP_LEFT_RIGHT":Image.FLIP_LEFT_RIGHT,
    "FLIP_TOP_BOTTOM":Image.FLIP_TOP_BOTTOM,
    "ROTATE_90":Image.ROTATE_90,
    "ROTATE_180":Image.ROTATE_180,
    "ROTATE_270":Image.ROTATE_270,
    "TRANSPOSE":Image.TRANSPOSE,
    "TRANSVERSE":Image.TRANSVERSE
}
# for key,values in flip.items():
    # plt.figure(figsize=(10,10))
    # plt.subplot(1,2,1)
    # plt.imshow(image)
    # plt.title("original")
    # plt.subplot(1,2,2)
    # plt.imshow(image.transpose(values))
    # plt.title(key)
    # plt.show()

#CROPPING AN IMAGE

#USING CONVENTIONAL METHOD
xupper=150
xlower=400
yupper=150
ylower=400
crop_top=array[xupper:xlower,yupper:ylower,:]#[vertical,horizontal,color]
# plt.figure(figsize=(10,10))
# plt.imshow(crop_top)
# plt.show()


#USING CROP FROM PIL
left=150
upper=150
right=400
lower=400
image=Image.open("cat.png")
crop_image=image.crop((left,upper,right,lower))
# plt.figure(figsize=(10,10))
# plt.imshow(crop_image)
# plt.show()


#FLIPPING THE NEW IMAGE
crop_image_flip=crop_image.transpose(Image.FLIP_LEFT_RIGHT)
# crop_image_flip.show()


#CHANGING SPECIFIC PIXELS

#CONVENTIONAL METHOD
array_sq=np.copy(array)
array_sq[upper:lower,left:right,1:2]=0

# plt.figure(figsize=(5,5))
# plt.subplot(1,2,1)
# plt.imshow(array)
# plt.title("orignal")
# plt.subplot(1,2,2)
# plt.imshow(array_sq)
# plt.title("ALTERED IMAGE")
# plt.show()

#USING IMAGEDRAW FROM PIL
image_draw=image.copy()
image_fn=ImageDraw.Draw(im=image_draw)
# shape=[left,upper,right,lower]
# image_fn.rectangle(xy=shape,fill="black")
# plt.figure(figsize=(10,10))
# plt.imshow(image_draw)
# plt.show()

#USING OTHER SHAPES
image_fn.text(xy=(150,200),text="box",fill=(0,0,0))
# plt.figure(figsize=(10,10))
# plt.imshow(image_draw)
# plt.show()

#OVERLAYING ONE IMAGE OVER ANOTHER
image_lenaa=Image.open("lenna.png")
array_lenna=np.array(image_lenaa)
    #REASSINGING THE PIXEL VALUES
array_lenna[upper:lower,left:right,:]=array[upper:lower,left:right,:]
# plt.imshow(array_lenna)
# plt.show()

#USING THE PASTE METHOD TO OVERLAY ONE IMAGE ON ANOTHER
image_lenaa.paste(crop_image,box=(left,upper))
# plt.imshow(image_lenaa)
# plt.show()

image=Image.open("cat.png")
new_image=image
copy_image=image.copy()
# print(id(image),id(new_image),id(copy_image))
shape=(left,upper,right,lower)
image_fn=ImageDraw.Draw(im=image)
image_fn.text(xy=(0,0),text="box",fill=(0,0,0))
image_fn.rectangle(xy=shape,fill="red")
# plt.figure(figsize=(10,10))
# plt.subplot(121)
# plt.imshow(new_image)
# plt.subplot(122)
# plt.imshow(copy_image)
# plt.show()








#TRYING ANOTHER IMAGE FOR FLIP
image1=Image.open("baboon.png")
im_flip1=ImageOps.flip(image1)
im_mirror1=ImageOps.mirror(image1)
# plt.figure(figsize=(10,10))
# plt.subplot(121)
# plt.imshow(im_flip1)
# plt.subplot(122)
# plt.imshow(im_mirror1)
# plt.show()