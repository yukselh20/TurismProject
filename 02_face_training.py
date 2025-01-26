import cv2
import numpy as np
import os
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

def getImagesAndLabels(path):
    face_samples = []
    ids = []
    
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5) as face_detection:

        for image_path in [os.path.join(path,f) for f in os.listdir(path)]:
            try:
                img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if img_gray is None:
                    continue
                    
                if img_gray.shape[0] < 50 or img_gray.shape[1] < 50:
                    continue

                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                results = face_detection.process(img_rgb)

                if results.detections:
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw = img_gray.shape[:2]
                    
                    x = max(0, int(bboxC.xmin * iw))
                    y = max(0, int(bboxC.ymin * ih))
                    w = min(iw - x, int(bboxC.width * iw))
                    h = min(ih - y, int(bboxC.height * ih))
                    
                    if w > 10 and h > 10:
                        face_roi = img_gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (100, 100))
                        
                        # Ek kanal kontrolü
                        if len(face_roi.shape) > 2:
                            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                            
                        face_samples.append(face_roi)
                        ids.append(int(os.path.split(image_path)[-1].split(".")[1]))
                        
            except:
                continue

    return face_samples, np.array(ids)

print("\n [INFO] Training faces...")
faces, ids = getImagesAndLabels('dataset')

# Boş veri kontrolü
if len(faces) == 0:
    raise ValueError("No valid faces found in dataset!")

print(f"Training with {len(faces)} valid face samples")

# LBPH parametre optimizasyonu
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2, 
    neighbors=16,
    grid_x=8,
    grid_y=8
)

recognizer.train(faces, ids)
recognizer.write('trainer.yml')
print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")