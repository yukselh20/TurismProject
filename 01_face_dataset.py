import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Müşteri bilgileri alma kısmı aynı kalacak
id = 1
ad = input("Müşterinin adını giriniz: ")
soyad = input("Müşterinin soyadını giriniz: ")
tcNo = input("Müşterinin Tc numarasını giriniz: ")

with open("musteriler.txt","r") as dosya:
    musterilerListesi = dosya.readlines()

if len(musterilerListesi) == 0:
    id = 1
else:
    with open("musteriler.txt","r") as dosya:
        id = int(musterilerListesi[-1].split(")")[0]) + 1
        
with open("musteriler.txt","a+") as dosya:
    dosya.write("{}){}-{}-{}\n".format(id,ad,soyad,tcNo))

face_id = id

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")


with mp_face_detection.FaceDetection(
    model_selection=1, 
    min_detection_confidence=0.7) as face_detection:

    count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            break

        # MediaPipe için RGB dönüşümü
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                # Yüz sınırlayıcı kutu koordinatları
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Yüz kırpma ve kaydetme
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_roi = gray[y:y+h, x:x+w]
                
                cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", face_roi)
                count += 1
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow('image', img)
        if cv2.waitKey(1) == 27 or count >= 100:
            break

cam.release()
cv2.destroyAllWindows()