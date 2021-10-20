import cv2
import numpy as np
import matplotlib.pyplot as plt

yolo = cv2.dnn.readNet('Resources/yolov3-tiny.weights', 'Resources/yolov3-tiny.cfg')

classes = []
with open('Resources/coco.names') as f:
    classes = f.read().splitlines()

img = cv2.imread('Resources/objdet.jpg')
#img = cv2.resize(img, (600, 600))
print(img.shape)
blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB=True, crop=False)

yolo.setInput(blob)

output_layer_names = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layer_names)

boxes = []
confidences = []
class_ids = []
width=height=w=h=1

for output in layeroutput:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.7 :
            center_x = int(detection[0]* width)
            center_x = int(detection[0] * height)
            w = int(detection[0] * width)
            h = int(detection[0] * height)

            x = int(center_x - w / 2)
            y = int(center_x - h / 2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_COMPLEX
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
    x,y,w,h=boxes[i]
    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i]))
    color = colors[i]

    cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
    cv2.putText(img, label+" "+confi, (x,y+20), font, 1, (0,255,0),2)

cv2.imshow("img", img)

cv2.waitKey(0)