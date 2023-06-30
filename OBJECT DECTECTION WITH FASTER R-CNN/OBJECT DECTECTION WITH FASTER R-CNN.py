#OBJECT DECTECTION WITH FASTER R-CNN
#DEEP LEARNING LIBRARIES
import torchvision
from torchvision import transforms
import torch
from torch import no_grad


#USED TO GET DATA FROM WEB
import requests

#LIBRARY FOR IMAGE PROCESSING AND VISUALIZATION
import  cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#THIS FUNCTION WILL ASSIGN A STRING NAME TOA PREDICTED CLASS AND ELIMINATE PREDICTIONS WHOSE LIKELIHOOD IS UNDER THRESHOLD

def get_predictions(pred,threshold=0.8,objects=None):
    #pred : alist where each element contains a tuple that corresponds to information about the diffrent objects ; each elements includes a tuple  with the class yhat , probabilityof belonging to that class and coordinates of the bounding box corresponding to the object
    #image: frozen surface
    #predicte_classes:a list where each element contains a tuple that corresponds to information abou the diffrent objects ; each e;ement incudes a tuple with the class ame, probability of belonging to that class and the coordinates of the bounding box corresponding to the object
    predicted_classes= [(COCO_INSTANCE_CATEGORY_NAMES[i],p,[(box[0], box[1]), (box[2], box[3])]) for i,p,box in zip(list(pred[0]['labels'].numpy()),pred[0]['scores'].detach().numpy(),list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes=[  stuff  for stuff in predicted_classes  if stuff[1]>threshold ]
    if objects  and predicted_classes :
        predicted_classes=[ (name, p, box) for name, p, box in predicted_classes if name in  objects ]
    return predicted_classes

#this function draws box around each object
def draw_box(predicted_classes, image, rect_th=10, text_size=3, text_th=3):
    #predicted class :a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object
    #image : frozen surface
    img = (np.clip (
        cv2.cvtColor ( np.clip ( image.numpy ( ).transpose ( (1 , 2 , 0) ) , 0 , 1 ) , cv2.COLOR_RGB2BGR ) , 0 ,
        1 ) * 255).astype ( np.uint8 ).copy ( )
    for predicted_class in predicted_classes :
        label = predicted_class [ 0 ]
        probability = predicted_class [ 1 ]
        box = predicted_class [ 2 ]
        pt1 = (int ( box [ 0 ] [ 0 ] ) , int ( box [ 0 ] [ 1 ] ))
        pt2 = (int ( box [ 1 ] [ 0 ] ) , int ( box [ 1 ] [ 1 ] ))
        cv2.rectangle ( img , pt1 , pt2 , (0 , 255 , 0) , rect_th )
        cv2.putText ( img , label , pt1 , cv2.FONT_HERSHEY_SIMPLEX , text_size , (0 , 255 , 0) , thickness = text_th )
        cv2.putText ( img , label + ": " + str ( round ( probability , 2 ) ) , pt1 , cv2.FONT_HERSHEY_SIMPLEX ,
                      text_size , (0 , 255 , 0) , thickness = text_th )
    plt.imshow ( cv2.cvtColor ( img , cv2.COLOR_BGR2RGB ) )
    plt.show ( )
    del img
    del image

#THIS FUNCTION WILL FREE UP SOME MEMORY
def save_RAM(image_=False):
    global image,img,pred
    torch.cuda.empty_cache()
    del(img)
    del(pred)
    if image_:
        image.close()
        del(image)


#LOAD PRE-TRAINED FASTER R-CNN
# Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image pre-trained on COCO.
model_=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_.eval()
for name,param in model_.named_parameters():
    param.requires_grad=False
print("done")
def model(x):
    with torch.no_grad():
        yhat=model_(x)
    return yhat
COCO_INSTANCE_CATEGORY_NAMES=[ '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# print(len(COCO_INSTANCE_CATEGORY_NAMES))


#object localization
img_path='jeff_hinton.png'
half=0.5
image=Image.open(img_path)
image.resize( [int(half * s) for s in image.size] )
# plt.imshow(image)
# plt.show()

#transform object for transforming object to a tensor
transform=transforms.Compose([transforms.ToTensor()])
#we convert the image to a tensor
img=transform(image)
#making a prediction the output is a dictionary with several predicted class, the probability of belonging to that class and the coordinates of the bounding box corresponding to that class
pred=model([img])
#we have 35 diffrent class predictions , ordered by likely hood scores for potential objects
print(pred[0]['labels'])
#we have likelihood of each class
print(pred[0]['scores'])
# the class number corressponds to the index of the list with the corresopnding category name
index=pred[0]['labels'][0].item()
print(COCO_INSTANCE_CATEGORY_NAMES[index])

#we have the coodinates of the bounding box
bounding_box=pred[0]['boxes'][0].tolist()
print(bounding_box)

# These components correspond to the top-left corner and bottom-right corner of the rectangle,more precisely :
#
# top (t),left (l),bottom(b),right (r)
#
# we need to round them
#












t,l,r,b=[round(x) for x in bounding_box]

#we convert the tensor to an open cv array and plot an image with the box
img_plot=(np.clip(cv2.cvtColor(np.clip(img.numpy().transpose((1,2,0)),0,1),cv2.COLOR_BGR2RGB),0,1)*255).astype(np.int8)
cv2.rectangle(img_plot,(t,l),(r,b),(0,255,0),10)#draw rectangle with the coordinates
# plt.show()
del img_plot,t,l,r,b

#we can localize objects ; we do this using the  function get _predictions . the input is the predictions pred and the pbjects you would like to localize
pred_class=get_predictions(pred,objects='person')
# draw_box(pred_class,img)
del pred_class


#we can set a threshold . here we set the threshold 1i.e here we set the threshold 1 i.e100% likelihood
get_predictions(pred,threshold = 1,objects='person')
#here we have no output as the likelihood is not 100% lets try a threshold of 0.98 and use the function  draw_box to draw_box to draw the box  and plot the class and its rounded likelihood
pred_thresh=get_predictions(pred,threshold = 0.98,objects="person")
# draw_box(pred_thresh,img)
del pred_thresh

#delete objects to save memory, we will run afterevery cell
save_RAM(image_ =True)

#wecan locate multiple bject , consider the following image ,we can detect the people in the image
img_path='DLguys.jpeg'
image=Image.open(img_path)
image.resize([int(half*s) for s in image.size])
# plt.imshow(np.array(image))
# plt.show()

#we can set a threshold to dectect the object , 0.9 seems to work
img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.8,)
# draw_box(pred_thresh,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_thresh

#or we can use objects parameter
pred_obj=get_predictions(pred,objects="person")
# draw_box(pred_obj,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_obj


#If we set the threshold too low, we will detect objects that are not there.
pred_thresh=get_predictions(pred,threshold=0.01)
# draw_box(pred_thresh,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_thresh

# the following lines will speed up your code by using less RAM.
save_RAM(image_=True)


#OBJECT DECTECTIONS
img_path='istockphoto-187786732-612x612.jpeg'
image=Image.open(img_path)
image.resize([int(half*s) for s in image.size])
# plt.imshow(np.array(image))
# plt.show()
del img_path


#if we set a threshold we can detect all onjects whose likelihood is above that threshold
img=transform(image)
pred=model([img])
pred_thresh=get_predictions(pred,threshold = 0.97)
# draw_box(pred_thresh,img,rect_th = 1,text_size = 1,text_th = 1)
del pred_thresh


save_RAM(image_=True)
#we can specify the objects we wouldlike to classify  for  example cats and dogs
img_path='istockphoto-187786732-612x612.jpeg'
image=Image.open(img_path)
img=transform(image)
pred=model([img])
pred_obj=get_predictions(pred,objects = ['dog','cat'])
# draw_box(pred_obj,img,rect_th = 1,text_size = 0.5,text_th = 1)
del pred_obj

#if we set the treshold too low we may detectobject with a lowlikelihood of being correct, here we set the threshold to0.7 and we incorrectly detect a cat
pred_thresh=get_predictions(pred,threshold = 0.70,objects = ['dog','cat'])
# draw_box(pred_thresh,img,rect_th = 1,text_size = 1,text_th = 1)
del pred_thresh
save_RAM(image_=True)


img_path='watts_photos2758112663727581126637_b5d4d192d4_b.jpeg'
image=Image.open(img_path)
image.resize([int(half*s) for s in image.size])
# plt.imshow(np.array(image))
# plt.show()
del img_path


img=transform(image)
pred=model([img])
pred_thresh=get_predictions(pred,threshold = 0.997)
# draw_box(pred_thresh,img)
del pred_thresh
save_RAM(image_ = True)

#TEST MODEL WITH AN UPLODED IMAGE
url='https://www.plastform.ca/wp-content/themes/plastform/images/slider-image-2.jpg'
image=Image.open(requests.get(url,stream = True).raw).convert("RGB")
del url
img=transform(image)
pred=model([img])
pred_thresh=get_predictions(pred,threshold = 0.95)
draw_box(pred_thresh,img)
del pred_thresh
save_RAM(image_ = True)


