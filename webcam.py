import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam introuvable")
    raise SystemExit

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    cv2.imshow("AirWriter - Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()