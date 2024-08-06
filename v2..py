import cv2
import numpy as np
import os
import sys
from collections import deque
import face_recognition
from deepface import DeepFace
import google.generativeai as genai
from dotenv import load_dotenv
import threading
import time
import requests
import RPi.GPIO as GPIO

# Load environment variables
load_dotenv()

mobile_camera_url = 'https://192.168.1.35:8080/shot.jpg'

# Configure generative AI
genai.configure(api_key=os.getenv('API_KEY'))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="start a coversation",
)

history = []

def gen_ai(user_input):
    print('Generating response...')
    chat_session = model.start_chat(history=history)
    response = chat_session.send_message(user_input)
    model_response = response.text
    history.append({"role": "user", "parts": [user_input]})
    history.append({"role": "model", "parts": [model_response]})
    return model_response

class AIAssistant:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.current_face = None
        self.in_conversation = False

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.encode_faces()

        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        self.red_led = 17
        self.green_led = 27
        self.white_led = 22
        GPIO.setup(self.red_led, GPIO.OUT)
        GPIO.setup(self.green_led, GPIO.OUT)
        GPIO.setup(self.white_led, GPIO.OUT)

        # Turn on red LED at startup
        self.set_led(self.red_led, True)

    def encode_faces(self):
        for image in os.listdir('features/faces'):
            face_image = face_recognition.load_image_file(f'features/faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

    def detect_face(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def recognize_face(self, frame):
        rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = 'Unknown'
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            return name.split('.')[0]
        return 'Unknown'

    def detect_emotion(self, face_roi):
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']

    def run(self):
        cap = cv2.VideoCapture(mobile_camera_url)

        if not cap.isOpened():
            sys.exit("Cannot open camera")

        while True:
            raw = requests.get(mobile_camera_url, verify=False)
            img = np.array(bytearray(raw.content), dtype=np.uint8)

            frame = cv2.imdecode(img, -1)

            if not self.in_conversation:
                self.set_led(self.green_led, True)
                faces = self.detect_face(frame)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]  # Only consider the first detected face
                    face_roi = frame[y:y + h, x:x + w]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    name = self.recognize_face(face_roi)
                    emotion = self.detect_emotion(face_roi)
                    if name != 'Unknown':
                        self.current_face = (name, emotion)
                        threading.Thread(target=self.start_conversation, args=(name, emotion)).start()
                        self.in_conversation = True
                        self.set_led(self.green_led, False)
                        self.set_led(self.white_led, True)
            else:
                cv2.putText(frame, f"Conversing with {self.current_face[0]} ({self.current_face[1]})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("AI Assistant", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

    def set_led(self, led, state):
        GPIO.output(led, state)

    def start_conversation(self, name, emotion):
        print(f"Starting conversation with {name}, who is {emotion}")
        user_input = f"{name} is feeling {emotion}. Start a conversation."
        response = gen_ai(user_input)
        print(f"AI Response: {response}")
        time.sleep(10)  # Simulating a conversation time
        self.in_conversation = False
        self.set_led(self.white_led, False)
        self.set_led(self.red_led, True)

if __name__ == "__main__":
    assistant = AIAssistant()
    assistant.run()
