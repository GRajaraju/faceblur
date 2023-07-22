import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
print(face_cascade)
cap = cv2.VideoCapture(0)
scaling_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor,
                       fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_bb = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x,y,w,h) in face_bb:
        print(x,y,w,h)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('Face', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

