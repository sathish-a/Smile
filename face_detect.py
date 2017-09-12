import cv2
from scipy.ndimage import zoom
import numpy as np

def detect_face(frame):
    cascPath = "/Users/sathish/anaconda/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return gray, detected_faces


def extract_face_features(grays, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = offset_coefficients[0] * w
    vertical_offset = offset_coefficients[1] * h
    st1 = int(y + vertical_offset)
    end1 = int(y + h)
    st2 = int(x + horizontal_offset)
    end2 = int(x - horizontal_offset + w)
    extracted_face = grays[st1:end1, st2: end2]

    new_extracted_face = zoom(extracted_face, (112. / extracted_face.shape[0], 92. / extracted_face.shape[1]))
    new_extracted_face = np.array(new_extracted_face, dtype=np.uint8)
    return new_extracted_face


cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # detect faces
    gray, detected_faces = detect_face(frame)

    face_index = 0

    # predict output
    for face in detected_faces:
        (x, y, w, h) = face

        if w > 100:
            # draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # extract features
            processed_extracted_face = extract_face_features(gray, face, (0.075, 0.05))  # (0.075, 0.05) (0.03, 0.05)

            # predict smile
            # prediction_result = predict_face_is_smiling(extracted_face)

            # draw extracted face in the top right corner
            frame[face_index * 64: (face_index + 1) * 112, -93:-1, :] = cv2.cvtColor(processed_extracted_face, cv2.COLOR_GRAY2RGB)

            # annotate main image with a label
            # if prediction_result == 1:
            #     cv2.putText(frame, "SMILING", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
            # else:
            #     cv2.putText(frame, "not smiling", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)

            # increment counter
            face_index += 1


    # Display the resulting frame

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
