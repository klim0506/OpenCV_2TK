import time
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fps = 0
start_time = time.time()

out = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width, frame_height))

cap.set(4, 640) #cap.set(3, 640)
cap.set(3, 480) #cap.set(4, 480)

while True:
    _, img = cap.read()
    _, img = cap.read()

    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results = model.predict(img)

    for r in results:

        annotator = Annotator(img)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    img = annotator.result()
    cv2.imshow('YOhttps://habr.com/ru/articles/593547/LO V8 Detection', img)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
    out.write(img)

    fps = round(1 / (time.time() - start_time))
    start_time = time.time()
    print(fps)

out.release()
cap.release()
cv2.destroyAllWindows()