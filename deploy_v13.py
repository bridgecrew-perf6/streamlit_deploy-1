# Import dependencies
import streamlit as st
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from keras.models import load_model
from playsound import playsound

# Define function for mediapipe detection and drawing landmark connections on video feed"""
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

st.title('AI Personal Trainer')

@st.cache  # cache the image recognition model
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width / float(w)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = width, int(h * r)

    # resize image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    return np.array([pose])


# """ ABOUT APP"""
app_mode = st.sidebar.selectbox("Choose the App Mode", ['About App', 'Squat', 'Boxing'])

if app_mode == 'About App':
    st.markdown("Provide Recommendation for **1 Full Set of 10 Squats**.")
    st.image("team_icon.JPG", caption='Team Icon', width=250)
    st.markdown('#')

    st.image("mediapipe.png", width=250)
    st.markdown("Supported by MediaPipe in extracting body coordinates.")
    st.markdown('#')
    st.markdown('#')

    st.image("discussion.JPG", width=480)  # change to appropriate path
    st.markdown("Capstone Project presented by Xccelerate Data Science Team A:")
    st.markdown("**Nicholas**, **Hector**, **Shayan**, **Jennifer** on 18 Feb 2022.")

# """ VIDEO INPUT FOR SQUAT PREDICTION"""
elif app_mode == 'Squat':

    # Webcam controls
    st.sidebar.subheader("Webcam Controls")

    webcam = st.sidebar.radio("Choose your Webcam input", ('Internal', 'External'))

    wc = 0 if webcam == 'Internal' else 1

    start = st.sidebar.button("Start")

    st.sidebar.subheader("Model Controls")

    max_reps = st.sidebar.number_input("Max Reps to train", min_value=5, max_value=50, value= 10, step=5)

    # confidence interval controls
    detection_confidence = st.sidebar.slider("Min Detection Confidence", min_value=0.1, max_value=1.0, value=0.7)
    tracking_confidence = st.sidebar.slider("Min Tracking Confidence", min_value=0.1, max_value=1.0, value=0.7)

    # set rest time
    restFrames = st.sidebar.number_input("Resting Frames between Squats", min_value=10, max_value=100, value=20)

    n_features = 132
    n_steps = 2
    n_length = 20

    # Actions that we try to detect
    actions = np.array(['Walk', 'Standing', 'Good Squat',
                        'Shallow Squat Half',
                        'Both Knee Valgus (up and down)',
                        'Good Morning Squat'])

    model = load_model('squat_test_JSN_15Feb_2by20_70.h5')

    sequence = []
    exercise_pred = ''
    prediction = []

    if start:
        stframe = st.empty()

        # get our input video here
        cap = cv2.VideoCapture(wc)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.markdown("**No. of Reps**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Prediction**")
            kpi2_text = st.markdown("0")

        # status_text = st.empty()  # print corresponding message for each predicted squat

        # Squat prediction
        with mp_pose.Pose(min_detection_confidence=detection_confidence,
                          min_tracking_confidence=tracking_confidence) as pose:

            while cap.isOpened() and len(prediction) < max_reps:

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, pose)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                #########################################################################################

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                if len(sequence) == restFrames - 3:
                    playsound("start.mp3")

                if restFrames - 10 < len(sequence) < restFrames:
                    cv2.putText(image, "START!", (550, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4,
                                cv2.LINE_AA)

                if len(sequence) == 40 + restFrames:
                    res = model.predict(np.array(sequence[-40:]).reshape(1, n_steps, n_length, n_features))[0]
                    sequence = []
                    prediction.append(actions[np.argmax(res)])
                    exercise_pred = actions[np.argmax(res)]
                    playsound(f"message{np.argmax(res)}.mp3")

                # Show to screen
                kpi1_text.write(f"<h3 style='text-align: left; color:red;'>{len(prediction)}</h3>",
                                unsafe_allow_html=True)
                kpi2_text.write(f"<h3 style='text-align: left; color:red;'>{exercise_pred}</h3>",
                                unsafe_allow_html=True)

                image = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
                image = image_resize(image=image, width=640)
                stframe.image(image, channels='BGR', use_column_width=True)

            # count poses in prediction and save to dict
            squat_dict = {squat: prediction.count(squat) for squat in prediction}
            squat_dict_wo_walk = {squat: prediction.count(squat) for squat in prediction if squat not in ['Walk','Standing']}

            # summary recommendation logic
            squat_counts = squat_dict_wo_walk.values()
            max_value = max(squat_counts)

            max_preds = []
            for squat_pose, count in squat_dict_wo_walk.items():
                if count == max_value:
                    max_preds.append(squat_pose)  # add the max prediction into this list

            summary = pd.DataFrame.from_dict(squat_dict, orient='index', columns=['Count'])

            st.subheader("Summary of Reps:")

            # Display the summary table
            st.dataframe(summary)

            ####################################### Good Squat ##########################################

            if "Good Squat" in max_preds:
                st.markdown("Great job on those squats! ")
                st.markdown("Here are some training recommendations on how you can take your squat to the next level! ")
                st.markdown("**Barbell Squats: **")
                st.video("https://www.youtube.com/watch?v=SW_C1A-rejs&ab_channel=ScottHermanFitness")
                st.markdown(
                    "With the barbell placed on your back, perform the same squat movement like you did just now! Start with a lighter weight, and slowly increase the weight session after session.")
                st.markdown("Try to squat 2-3 times a week, for 4 sets of 10 reps.")
                st.markdown("**Goblet Squat:**")
                st.video("https://www.youtube.com/watch?v=gCESNsDsbqk&ab_channel=BuiltLean")
                st.markdown(
                    "Grab a dumbbell and cup the head of the dumbbell with the palms of your hands, and placed in front of your chest. Keep your head up, and squat down by keeping your knees out and over your knees. Perform the same squat movement like you did just now! Start with a lighter weight, and slowly increase the weight session after session.")
                st.markdown("Try to squat 2-3 times a week, for 4 sets of 10 reps. ")
            else:
                st.write(
                    "Nice job! We can still make some improvements in your squat form. Here are some suggestions!")

            ################################## Shallow Squat Half #########################################

            if "Shallow Squat Half" in max_preds:
                st.markdown("**Heel Elevated Squat:**")
                st.video("https://youtu.be/hqh91-tY7Ss?t=43")
                st.markdown(
                    "Part of the reason why it's difficult to squat down is because our ankles are too tight. To quickly fix this, elevate your heels by 0.5 to 1 inch off the ground with a plate or flat object, while keeping your toes on the ground. Perform the same squat movement as you did just now, with more intention on squatting lower with each rep!")
                st.markdown("#")
                st.markdown("Do these every time you squat!")
                st.markdown("**Banded Ankle Mobilization:**")
                st.video("https://youtu.be/IikP_teeLkI?t=219")
                st.markdown(
                    "Over the long term, we want to get your ankles flexible enough to squat down to the bottom without assistance. First, grab a big strong band. Wrap one end of the band to a strong anchor (e.g. a door or pole) and the wrap the other end around your ankle. Kneel in a lunge position, and push your knee over your toes to stretch the ankle. See video for more explanation! ")
                st.markdown("Try to do this 3-5 times a week, 10-15 reps on each ankle.")

            ################################ Both Knee Valgus (up and down) ################################

            if "Both Knee Valgus (up and down)" in max_preds:
                st.markdown("**Single Leg Squat:**")
                st.video("https://youtu.be/2C-uNgKwPLE?t=57")
                st.markdown(
                    "Start the exercise standing with your front foot on the ground, and your rear foot on a chair or stool. Your front leg will be the working leg during the exercises, and your back leg will be used for balance. Once in position, lower yourself into a deep squat position, while keeping your chest up. Return the starting position by pushing the heel of your front foot. Majority of the weight should be on your front leg.")
                st.markdown("Try to do these 2 times a week, 5-8 reps on each leg.")
                st.markdown("**Banded Squats**")
                st.video("https://youtu.be/9CbPyDr2P0w?t=18")
                st.markdown(
                    "With a band, wrap the band around your knees. Keep you feet around shoulder width apart or slightly wider, depending on what you’re comfortable with. As you perform your squats, keep your knees out and resist the band from pushing your knees inward. You should feel some attention around your glutes.")
                st.markdown("Try to do these 3 times a week, for 10 reps. ")

            ############################### Good Morning Squat ######################################

            if "Good Morning Squat" in max_preds:
                st.markdown("**Keep your Core Tight:**")
                st.video("https://youtu.be/PJX1CyjbMic?t=34")
                st.markdown(
                    "Before starting your squat, take a deep breath with your nose and mouth, and feel like you're putting the air into your stomach. Expand your stomach outward and hold the air in that general area. Almost imagine like you're tightening your core right before someone tries to punch you in the stomach. While holding your breathe, begin your squat.")
                st.markdown("Practice this technique right before your squat workout for 10 reps.")
                st.markdown("**Keep your Chest up:**")
                st.markdown(
                    "While squatting, make sure to keep your chest up and facing forward. Imagine you are squatting right in front of a wall, you want your chest facing the wall rather than facing the floor every time you squat.")
                st.markdown("Practice this technique right before your squat workout for 10 reps. ")


