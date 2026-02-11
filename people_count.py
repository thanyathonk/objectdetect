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


cap = cv2.VideoCapture("./video/people.mp4")
mask = cv2.imread("./image/mask-1.png")

# Tracking
Tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []

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
    
    imgGraphics = cv2.imread('./image/graphics-1.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730,260))

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
                currentClass == "person"
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
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = Tracker.update(detections)
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=3, colorR=(255, 0, 0))
        cvzone.putTextRect(
            img,
            f"{id}",
            (max(0, x1), max(35, y1)),
            scale=0.6,
            thickness=1,
            offset=3,
        )

        # point for counting
        cx, cy = x1 + w // 2, y1 + w // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            # totalCount += 1
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0,255,0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            # totalCount += 1
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0,255,0), 5)

    # cvzone.putTextRect(img,f"Count: {len(totalCount)}",(50, 50),)
    cv2.putText(img, f'{len(totalCountUp)}', (929,345), cv2.FONT_HERSHEY_PLAIN, 5, (139,195,75), 7)
    cv2.putText(img, f'{len(totalCountDown)}', (1191,345), cv2.FONT_HERSHEY_PLAIN, 5, (50,50,230), 7)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
