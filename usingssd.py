import cv2

# paths
configPath = 'Resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'Resources/frozen_inference_graph.pb'
imgPath = 'Resources/objdet.jpg'

img = cv2.imread(imgPath)

classes = []
with open('Resources/coco.names') as f:
    classes = f.read().splitlines()

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


class_ids, confidences, boxes = net.detect(img,confThreshold=0.5)

font = cv2.FONT_HERSHEY_SIMPLEX
for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
    if class_id < 82:
        cv2.rectangle(img, box, (255, 0, 0), 1)
        cv2.putText(img, classes[class_id - 1], (box[0]+10,box[1]-10), font, 0.5, (255,0,0),1)

cv2.imshow("Image", img)
cv2.waitKey(0)