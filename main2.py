from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

model = YOLO(r"runs/detect/train11/weights/best.pt")
# model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while True:
    ret, img = cap.read()

    if not ret:  # если не получили картинку
        continue

    results = model.predict(img)

    for r in results:

        annotator = Annotator(img)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    img = annotator.result()
    cv2.imshow('Video', img)

    k = cv2.waitKey(10)

    if k == ord(' '):
        break

cap.release()




