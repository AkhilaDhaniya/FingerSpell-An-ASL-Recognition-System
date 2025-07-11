import streamlit as st
import numpy as np
import tensorflow as tf
import cv2 as cv
import mediapipe as mp
import copy
import itertools
import keyboard  # To detect key press
import pyttsx3
# Sidebar Navigation with Custom Styling
st.sidebar.markdown("""
    <style>
    .sidebar-content {
        
        background-color:#FEBE10;
        border-radius: 10px;
    }
    .st-emotion-cache-1wqrzgstreamlit unl {
    position: relative;
    top: 0.125rem;
    background-color: #febe10;
    z-index: 999991;
    min-width: 244px;
    max-width: 550px;
    transform: none;
    border-radius: 9px;
    }
    .app-title {
        color: white;
        font-size: 35px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    <div class="st-emotion-cache-1wqrzgl">
        <div class="app-title">FingerSpell</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar sections
pages = ["Home" , "About", "Translator"]
selected_page = None
for page in pages:
    if st.sidebar.button(page):
        selected_page = page

if selected_page == "Home":
    st.title("Welcome to FingerSpell")
    st.write("Navigate through the sidebar to explore the app.")


    st.markdown("<h3 style='color: #DA70D6;'>Our Mission and Vision</h3>", unsafe_allow_html=True)
    st.write("Our mission is to create accessible and effective tools to help bridge the gap between hearing and non-hearing communities.Our mission is "
             "to leverage innovative technology and machine learning to create accessible solutions that bridge communication gaps, empower individuals, and enhance inclusivity. "
             "By developing tools like our ASL sign language recognition system, we aim to provide seamless and real-time translation of sign language, enabling more effective "
             "communication for the deaf and hard-of-hearing community. Through continuous learning and improvement, we strive to build systems that are not only functional "
             "but also user-friendly,helping create a more inclusive world where everyone can connect and interact without barriers.")

elif selected_page == "About":
    st.title("About FingerSpell")
    st.markdown("<h3 style='color: #DA70D6;'>Meet the Minds Behind the Magic </h3>", unsafe_allow_html=True)
    st.write("FingerSpell is an AI-powered ASL hand gesture recognition system that translates sign language into letters in real-time.")
    st.write("FingerSpell is developed by Akhila Dhaniya and Janeetta Agnes, final-year Computer Science students passionate about artificial intelligence and accessibility."
             " What started as an academic project became a mission to enhance sign language recognition through technology.")
    st.subheader("Technology Used:")
    st.write("FingerSpell is powered by a combination of machine learning and computer vision technologies:")

    st.write("1.TensorFlow & Keras – Used to train a deep learning model, achieving 74% accuracy 📊")
    st.write("2.MediaPipe Hands – Detects and tracks hand landmarks in real time ")
    st.write("3.OpenCV – Integrates camera functionality for gesture recognition ")
    st.write("4.Streamlit – Provides an interactive and user-friendly web interface ")
    st.write("5.Python – The core programming language that brings it all together")

elif selected_page == "Translator":
    st.title("Let's Translate")
    st.write("Hand Sign Recognition (ASL Translator)")

    def load_labels(label_path="model/keypoint_classifier/keypoint_classifier_label.csv"):
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    labels = load_labels()
    st.image("The-26-hand-signs-of-the-ASL-Language.png", caption="The 26 hand signs of the ASL Language",
             use_container_width=True)
    @st.cache_resource
    def load_tflite_model(model_path="model/keypoint_classifier/keypoint_classifier.tflite"):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details

    def predict_tflite(interpreter, input_details, output_details, data, labels):
        data = np.expand_dims(data, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output)
        return labels[predicted_index] if predicted_index < len(labels) else "Unknown", output

    def pre_process_landmark(landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
        for index in range(len(temp_landmark_list)):
            temp_landmark_list[index][0] -= base_x
            temp_landmark_list[index][1] -= base_y
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(map(abs, temp_landmark_list))
        return [n / max_value for n in temp_landmark_list]

    model_path = "model/keypoint_classifier/keypoint_classifier.tflite"
    interpreter, input_details, output_details = load_tflite_model(model_path)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    stframe = st.empty()
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    def calc_bounding_rect(frame, hand_landmarks):
        image_width, image_height = frame.shape[1], frame.shape[0]
        landmark_array = np.array([[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in hand_landmarks.landmark])
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def draw_bounding_rect(frame, brect):
        cv.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 2)
        return frame

    def draw_landmarks(frame, landmark_list):
        for landmark in landmark_list:
            x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
            cv.circle(frame, (x, y), 5, (255, 255, 255), -1)
            cv.circle(frame, (x, y), 5, (0, 0, 0), 1)
        return frame


    def text_to_speech(predicted_word):
        engine = pyttsx3.init()
        engine.say(predicted_word)
        engine.runAndWait()

    # Initialize placeholders for camera feed and predicted text
    camera_placeholder = st.empty()
    predicted_text_placeholder = st.empty()

    # Variables to store the predicted word
    predicted_word = ""
    confirmed_letter = ""  # Temporary storage for the latest detected letter

    # Setup the camera capture
    capture = cv.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            st.write("Error in capturing frame.")
            break

        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process hand landmarks (assuming `hands.process(frame_rgb)` is a pre-trained model)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = [[landmark.x, landmark.y] for landmark in hand_landmarks.landmark]
                features = pre_process_landmark(landmark_list)
                predicted_label, _ = predict_tflite(interpreter, input_details, output_details, features, labels)
                brect = calc_bounding_rect(frame, hand_landmarks)

                # Draw bounding box and display the predicted letter inside it
                x, y, w, h = brect
                cv.putText(frame, f"{confirmed_letter}", (x + 10, y - 10),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                frame = draw_bounding_rect(frame, brect)
                frame = draw_landmarks(frame, landmark_list)
               # Store the latest detected letter temporarily
            if predicted_label:
                confirmed_letter = predicted_label

        # Confirm Letter (Press 'S')
        if keyboard.is_pressed('s') and confirmed_letter:
            predicted_word += confirmed_letter
            confirmed_letter = ""  # Reset the temporary letter

        # Delete Last Letter (Press 'D')
        if keyboard.is_pressed('d') and predicted_word:
            predicted_word = predicted_word[:-1]

        # Add Space (Press 'Space')
        if keyboard.is_pressed('space'):
            predicted_word += " "

                # Reset Everything (Press 'R')
        if keyboard.is_pressed('r'):
            predicted_word = ""

        # Update the text display with the current predicted word
        predicted_text_placeholder.text(f"Predicted Word: {predicted_word}")
        if keyboard.is_pressed('enter') and predicted_word:
            text_to_speech(predicted_word)

        predicted_text_placeholder.text(f"Predicted Word: {predicted_word}")
        # Display the camera feed
        camera_placeholder.image(frame, channels="BGR", use_container_width=True)

    # Clean up after the loop
    capture.release()
    cv.destroyAllWindows()





