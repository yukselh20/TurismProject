import cv2
import mediapipe as mp
from time import time

# Ortak Ayarlar
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MediaPipe Ayarları
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(min_detection_confidence=0.75)

# Haar Cascade Ayarları
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    # MediaPipe ile Tespit
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mediapipe_results = face_detector.process(rgb_frame)
    
    # Haar Cascade ile Tespit
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Sonuçları Çiz
    # MediaPipe (Yeşil Kutu)
    if mediapipe_results.detections:
        for detection in mediapipe_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'MP: {detection.score[0]:.2f}', (x, y-10), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
    
    # Haar Cascade (Kırmızı Kutu)
    for (x, y, w, h) in haar_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Haar', (x, y-30), 
                   cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
    
    cv2.imshow('Karsilastirma: Yesil=MediaPipe, Kirmizi=Haar', frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()