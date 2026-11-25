import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict("ex2-26.png", conf=0.04)

# 入力画像
img = results[0].orig_img


boxes = results[0].boxes

count = 0
#print(results[0].names)
for box in boxes:
    # 物体領域のxy座標を取得する．
    if int(box.cls[0]) != 0:
        continue
    count += 1
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        thickness=3,
    )

print(count)

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()