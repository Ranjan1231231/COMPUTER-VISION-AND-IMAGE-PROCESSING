

import matplotlib.pyplot as plt
import os
import cv2
#DEFINING THE CONCATINATION OF TWO IMAGE
def get_conca_h(im1,im2):
    dst=Image.new('RGB',(im1.width+im2.width,im1.height))
    dst.paste(im1,(0,0))
    dst.paste(im2,(im1.width,0))
    return dst
my_image="lenna.png"
cwd=os.getcwd()
image_path=os.path.join(cwd,my_image)
image=cv2.imread(my_image)
# print(type(image))
# print(image.shape)
# print("max = {},min = {}".format(image.max(),image.min()))

#PLOTTING AN IMAGE
#THROUGH THE CV2 LIBRARY
# cv2.imshow("image",image)
# cv2.waitKey(100)
# cv2.destroyAllWindows()
#THROUGH THE PLT LIBRARY
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.show()#it will show diifrent coloured picture because opencv has BGR colour scheme insted of RGB
new_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10,10))
# plt.imshow(new_image)
# plt.show()

#LOADING IMAGE THROUGH ITS PATH
# image=cv2.imread(image_path)
# print(image.shape)

#SAVING IMAGE AS JPG FORMAT
# cv2.imwrite("lenna.jpg",image)

#GRAYSCALE IMAGES
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# print(image_gray.shape)
# plt.figure(figsize=(10,10))
# plt.imshow(image_gray,cmap='gray')
# plt.show()

#SAVING THE IMAGE AS A GREYSCALE IMAGE
# cv2.imwrite('lena_gray_cv.jpg',image_gray)


#LOADING A GREYSCALE IMAGE DIRECTLY
im_gray=cv2.imread("barbara.png",cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(10,10))
# plt.imshow(im_gray,cmap='gray')
# plt.show()

#COLOR CHANNELS
baboon=cv2.imread('baboon.png')
# plt.figure(figsize=(10,10))
# plt.imshow(cv2.cvtColor(baboon,cv2.COLOR_BGR2RGB))
# plt.show()

##ASSINING THE DIFFRENT RGB COLOURS
blue,green,red=baboon[:,:,0],baboon[:,:,1],baboon[:,:,2]

#CONCATINATING EACH IMAGE
im_bgr=cv2.vconcat([blue,green,red])
# plt.figure(figsize=(10,10))
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(baboon,cv2.COLOR_BGR2RGB))
# plt.title("RGB IMAGE")
# plt.subplot(122)
# plt.imshow(im_bgr,cmap='gray')
# plt.title("Diffrent color channels blue (top), green(middle),red(bottom)")
# plt.show()

#INDENXING
rows=256
# plt.figure(figsize=(10,10))
# plt.imshow(new_image[0:rows,:,:])
# plt.show()

columns=256
# plt.figure(figsize=(10,10))
# plt.imshow(new_image[:,0:columns,:])
# plt.show()


#COPYING THE IMAGE
A=new_image.copy()
# plt.imshow(A)
# plt.show()

# B = A
# A[:,:,:] = 0
# plt.imshow(B)
# plt.show()

#MANUPULATING THE ELEMENTS USING INDEXING

#keeping only red channel and convering everything else to 0
baboon_red=baboon.copy()
# baboon_red[:,:,0]=0
# baboon_red[:,:,1]=0
# plt.figure(figsize=(10,10))
# plt.imshow(cv2.cvtColor(baboon_red,cv2.COLOR_BGR2RGB))
# plt.show()

#keeping only blue channel and convering everything else to 0
baboon_blue=baboon.copy()
# baboon_blue[:,:,1]=0
# baboon_blue[:,:,2]=0
# plt.figure(figsize=(10,10))
# plt.imshow(cv2.cvtColor(baboon_blue,cv2.COLOR_BGR2RGB))
# plt.show()

#keeping only green channel and convering everything else to 0
baboon_green=baboon.copy()
# baboon_green[:,:,0]=0
# baboon_green[:,:,2]=0
# plt.figure(figsize=(10,10))
# plt.imshow(cv2.cvtColor(baboon_green,cv2.COLOR_BGR2RGB))
# plt.show()


image=cv2.imread('baboon.png')
# plt.figure(figsize=(10,10))
# plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# plt.show()

image_lenna=cv2.imread('lenna.png')
image_lenna_blue=image_lenna.copy()
image_lenna_blue[:,:,0]=0
image_lenna_blue[:,:,2]=0
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image_lenna_blue,cv2.COLOR_BGR2RGB))
plt.show()