import cv2
from ultralytics import YOLO

model = YOLO("best260408.pt")

#results = model.predict("ex2-26.png", conf=0.005, imgsz = 1920)

# 入力画像
img = cv2.imread("ex2-26.png")
#print(img.shape)

X0, Y0, X1, Y1 = 570, 240, 960, 440
S = 7

crop = img[Y0:Y1, X0:X1]  # [縦の範囲, 横の範囲] の順
crop = cv2.resize(crop, None, fx=S, fy=S)

results = model.predict(crop, conf=0.001)

#boxes = results[0].boxes

count = 0
#print(results[0].names)

for box in results[0].boxes:
    if int(box.cls[0]) != 0:
        continue
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    # 拡大を戻して、切り出した位置の分ずらす
    x1 = x1 // S + X0
    y1 = y1 // S + Y0
    x2 = x2 // S + X0
    y2 = y2 // S + Y0

    w = x2 - x1  # 枠の幅
    h = y2 - y1  # 枠の高さ

    if w < 9 or h < 9:  # 小さすぎる枠は飛ばす
        continue
    if w / h > 1.4 or h / w > 1.4:  # 横長・縦長すぎる枠は飛ばす
        continue

    count += 1
    print(count, w, h)  # 調整用に大きさを表示する

    cv2.rectangle(img, 
                  (x1, y1), 
                  (x2, y2), 
                  (0, 0, 255), 
                  thickness=3)
    cv2.putText(img, str(count), (x1, y1 - 5),     # 番号の文字と書く位置（枠の左上の少し上）
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,     # 字体と文字の大きさ
                (0, 0, 255), 1)      

print(count)
cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()