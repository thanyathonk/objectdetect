from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

model = YOLO("yolo_weight/yolov8n.pt")

classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


cap = cv2.VideoCapture("./video/cars.mp4")
mask = cv2.imread("./image/mask.png")

# Tracking
Tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Video ended or failed to read frame.")
        break

    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img, mask_resized)
    result = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # Bounding
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # confident
            conf = math.ceil((box.conf[0] * 100)) / 100
            currentClass = classNames[cls]

            if (
                currentClass == "car"
                or currentClass == "truck"
                or currentClass == "bus"
                or currentClass == "motobike"
                and conf > 0.3
            ):
                # cvzone.putTextRect(
                #     img,
                #     f"{classNames[cls]} {conf}",
                #     (max(0, x1), max(35, y1)),
                #     scale=0.6,
                #     thickness=1,
                #     offset=3,
                # )
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = Tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int,result)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=3, colorR=(255,0,0))
        cvzone.putTextRect(
            img,
            f"{id}",
            (max(0, x1), max(35, y1)),
            scale=0.6,
            thickness=1,
            offset=3,
        )

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
