import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

# Tanıyıcı ve veri yükleme
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# İsim listesi yükleme
with open("musteriler.txt","r") as dosya:
    musteriler = dosya.readlines()
names = ["Bilinmiyor"] + [eleman.split(")")[1].split("-")[0] for eleman in musteriler]

cam = cv2.VideoCapture(0)
cam.set(3, 1000)
cam.set(4, 800)

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5) as face_detection:  # Confidence threshold düşürüldü

    while True:
        ret, img = cam.read()
        if not ret:
            break

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                try:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw = img.shape[:2]
                    
                    # ROI koordinatlarını sınırla
                    x = max(0, int(bboxC.xmin * iw))
                    y = max(0, int(bboxC.ymin * ih))
                    w = min(iw - x, int(bboxC.width * iw))
                    h = min(ih - y, int(bboxC.height * ih))
                    
                    # Geçerli ROI kontrolü
                    if w > 0 and h > 0:
                        face_roi = img[y:y+h, x:x+w]
                        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        
                        # Yüzü yeniden boyutlandır (eğitimle aynı boyutta olmalı)
                        gray = cv2.resize(gray, (200, 200))  
                        
                        id, confidence = recognizer.predict(gray)
                        
                        # Güven eşiğini ayarla
                        if confidence > 70:  # Bu değeri veri setinize göre optimize edin
                            label = names[id]
                            color = (0,255,0)
                        else:
                            label = "Bilinmiyor"
                            color = (0,0,255)
                            
                        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
                        cv2.putText(img, f"{label} {confidence:.1f}%", 
                                   (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    else:
                        print("Geçersiz ROI boyutları")
                        
                except Exception as e:
                    print(f"Hata: {e}")

        cv2.imshow('Camera', img)
        if cv2.waitKey(1) in {27, ord('q')}:
            break

cam.release()
cv2.destroyAllWindows()