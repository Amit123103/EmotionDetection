import cv2
import numpy as np
import onnxruntime as ort
import os
import requests
from collections import deque, Counter

class EmotionDetector:
    def __init__(self):
        # 1. EMOTION MODEL (ONNX)
        self.emotion_model_path = "emotion-ferplus-8.onnx"
        self.emotions = ['neutral', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'contempt']
        
        # 2. AGE MODEL (Caffe)
        self.age_proto = "age_deploy.prototxt"
        self.age_model = "age_net.caffemodel"
        self.age_buckets = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        # 3. GENDER MODEL (Caffe)
        self.gender_proto = "gender_deploy.prototxt"
        self.gender_model = "gender_net.caffemodel"
        self.gender_list = ['Male', 'Female']

        # 4. FACE DETECTION MODEL (DNN - ResNet-10)
        self.face_proto = "deploy.prototxt"
        self.face_model = "res10_300x300_ssd_iter_140000.caffemodel"

        # 5. EYE CASCADE (Keep for eye detection)
        # Fix for Render/Headless: cv2.data.haarcascades might be invalid
        try:
            self.eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            if not os.path.exists(self.eye_cascade_path):
                 print(f"Warning: Cascade not found at {self.eye_cascade_path}. Checking local...")
                 self.eye_cascade_path = "haarcascade_eye.xml"
            
            if not os.path.exists(self.eye_cascade_path):
                 print("Downloading haarcascade_eye.xml...")
                 self.download_file("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml", "haarcascade_eye.xml")
                 self.eye_cascade_path = "haarcascade_eye.xml"

            self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade_path)
        except Exception as e:
            print(f"Error initializing Eye Cascade: {e}")
        
        self.age_net = None
        self.gender_net = None
        self.face_net = None
        self.ort_session = None

        # Stabilization History
        self.age_history = deque(maxlen=30) # Store last 30 frames (~1 sec)

        # Initialize
        print("Initializing EmotionDetector...")
        self.ensure_models_exist()
        self.load_models()

    def ensure_models_exist(self):
        # 1. EMOTION ONNX
        if not os.path.exists(self.emotion_model_path) or os.path.getsize(self.emotion_model_path) < 1000:
            print(f"Downloading Emotion Model to {self.emotion_model_path}...")
            self.download_file("https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx", self.emotion_model_path)

        # 2. AGE MODEL
        if not os.path.exists(self.age_proto) or os.path.getsize(self.age_proto) < 100:
            self.download_file("https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt", self.age_proto)
        if not os.path.exists(self.age_model) or os.path.getsize(self.age_model) < 1000000:
            self.download_file("https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel", self.age_model)

        # 3. GENDER MODEL
        if not os.path.exists(self.gender_proto) or os.path.getsize(self.gender_proto) < 100:
            self.download_file("https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt", self.gender_proto)
        if not os.path.exists(self.gender_model) or os.path.getsize(self.gender_model) < 1000000:
            self.download_file("https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel", self.gender_model)

        # 4. FACE MODEL (ResNet-10)
        if not os.path.exists(self.face_proto) or os.path.getsize(self.face_proto) < 100:
            self.download_file("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", self.face_proto)
        if not os.path.exists(self.face_model) or os.path.getsize(self.face_model) < 1000000:
            print("Downloading Advanced Face Model...")
            # Updated Link just in case
            self.download_file("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", self.face_model)

    def download_file(self, url, filename):
        print(f"Starting download: {filename} from {url}")
        try:
            response = requests.get(url, stream=True, allow_redirects=True)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                         if chunk: f.write(chunk)
                current_size = os.path.getsize(filename)
                print(f"Downloaded {filename} successfully ({current_size} bytes).")
            else:
                print(f"Failed to download {filename}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

    def load_models(self):
        # Load Emotion
        print(f"Loading Emotion Model from {self.emotion_model_path}...")
        try:
            if os.path.exists(self.emotion_model_path):
                self.ort_session = ort.InferenceSession(self.emotion_model_path)
                self.input_name = self.ort_session.get_inputs()[0].name
                print("Emotion Model Loaded Successfully.")
            else:
                print("Error: Emotion Model file not found!")
        except Exception as e:
            print(f"CRITICAL Error loading Emotion model: {e}")

        # Load Age
        print("Loading Age Model...")
        try:
             if os.path.exists(self.age_model) and os.path.exists(self.age_proto):
                self.age_net = cv2.dnn.readNet(self.age_model, self.age_proto)
                print("Age Model Loaded.")
             else:
                print("Error: Age Model files not found!")
        except Exception as e:
            print(f"Error loading Age model: {e}")

        # Load Gender
        print("Loading Gender Model...")
        try:
            if os.path.exists(self.gender_model) and os.path.exists(self.gender_proto):
                self.gender_net = cv2.dnn.readNet(self.gender_model, self.gender_proto)
                print("Gender Model Loaded.")
            else:
                print("Error: Gender Model files not found!")
        except Exception as e:
            print(f"Error loading Gender model: {e}")

        # Load Face
        print("Loading Face Model...")
        try:
            if os.path.exists(self.face_model) and os.path.exists(self.face_proto):
                self.face_net = cv2.dnn.readNet(self.face_model, self.face_proto)
                print("Face Model Loaded.")
            else:
                print("Error: Face Model files not found!")
        except Exception as e:
            print(f"Error loading Face model: {e}")

    def preprocess_emotion(self, face_img):
        face_img = cv2.resize(face_img, (64, 64))
        face_img = face_img.astype(np.float32)
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (64, 64))
        face_img = face_img[np.newaxis, np.newaxis, :, :] 
        return face_img

    def predict_age(self, face_img):
        if self.age_net is None: return "Unknown"
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        self.age_net.setInput(blob)
        preds = self.age_net.forward()
        i = preds[0].argmax()
        current_age = self.age_buckets[i]
        
        # Stabilization: Add to history and pick mode
        self.age_history.append(current_age)
        most_common = Counter(self.age_history).most_common(1)[0][0]
        return most_common
    
    def predict_gender(self, face_img):
        if self.gender_net is None: return "Unknown"
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        self.gender_net.setInput(blob)
        preds = self.gender_net.forward()
        i = preds[0].argmax()
        return self.gender_list[i]

    def generate_summary(self, emotion, age, gender, eyes_detected, speech_text):
        summary = f"The user is a {gender} (approx {age}), appearing {emotion}."
        
        # Emotion Nuance
        if emotion == 'sad':
             if not eyes_detected: summary = f"The {gender} user appears deeply sorrowful, possibly weeping."
             else: summary = f"The {gender} user seems sad or disappointed."
        elif emotion == 'happy':
            summary = f"The {gender} user is radiating happiness!"
        elif emotion == 'surprise':
            summary = "The user looks completely shocked!"
        elif emotion == 'anger':
            summary = "The user shows visible frustration."
            
        # Voice Context
        if speech_text:
            summary += f" They are saying: \"{speech_text}\""
            if "help" in speech_text.lower() or "sad" in speech_text.lower():
                summary += " (Voice indicates distress)."
        else:
            summary += " (User is silent)."

        return summary

    def detect_all(self, frame, speech_text=""):
        if self.ort_session is None or self.face_net is None:
            # print("Models not loaded, skipping detection")
            return []

        # DNN Face Detection
        h, w = frame.shape[:2]
        
        # Use simple blobFromImage
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        # Debug: Print max confidence to check if model is working
        if detections.shape[2] > 0:
            max_conf = np.max(detections[0, 0, :, 2])
            # if max_conf < 0.3:
               # Only print if no good detections to avoid spam
               # print(f"Low confidence face: {max_conf:.4f}")
        
        results = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Lower threshold to 0.3 to catch faces more easily
            if confidence > 0.3:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure within frame
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w-1, endX), min(h-1, endY)

                # Skip small faces
                if (endX - startX) < 20 or (endY - startY) < 20:
                    continue
                
                # Standardize variables for downstream compatibility
                x, y, w_box, h_box = startX, startY, endX - startX, endY - startY

                try:
                    roi_gray = frame[y:y+h_box, x:x+w_box]
                    roi_color = frame[y:y+h_box, x:x+w_box]
                    
                    if roi_gray.size == 0: continue

                    # 1. Detect Eyes (using Cascade on the detected face ROI)
                    # Use gray for eye detection
                    if len(roi_gray.shape) == 3:
                         roi_gray_eye = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
                    else:
                         roi_gray_eye = roi_gray
                    
                    eyes = []
                    if self.eye_cascade:
                         eyes = self.eye_cascade.detectMultiScale(roi_gray_eye)
                    
                    eye_boxes = [{"x": int(x+ex), "y": int(y+ey), "w": int(ew), "h": int(eh)} for (ex, ey, ew, eh) in eyes]

                    # 2. Predict Emotion
                    input_tensor = self.preprocess_emotion(roi_color) # Emotion model handles color/gray conversion internally
                    outputs = self.ort_session.run(None, {self.input_name: input_tensor})
                    emotion = self.emotions[np.argmax(outputs[0][0])]
                    
                    # 3. Predict Age
                    age = self.predict_age(roi_color)

                    # 4. Predict Gender
                    gender = self.predict_gender(roi_color)
                    
                    # 5. Generate Summary
                    summary = self.generate_summary(emotion, age, gender, len(eyes) > 0, speech_text)
                    
                    results.append({
                        "emotion": emotion,
                        "age": age,
                        "gender": gender,
                        "summary": summary,
                        "box": {"x": int(x), "y": int(y), "w": int(w_box), "h": int(h_box)},
                        "eyes": eye_boxes
                    })
                except Exception as e:
                    print(f"Error processing face: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        return results
