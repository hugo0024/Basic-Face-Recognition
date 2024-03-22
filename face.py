from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import face_recognition
import numpy as np

class TrackingObject:
    def __init__(self, center_position):
        self.center_positions = [center_position]
        self.counted = False

    def update(self, center_position):
        self.center_positions.append(center_position)
        
# Load the YOLOv8 model
model = YOLO('yolov8n-face.pt')

video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

base_dir = 'face_snapshots'
os.makedirs(base_dir, exist_ok=True)

# Set video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

fixed_screenshot_size = (120, 120)
timestamp_height = 20

# Keep track of unique persons and their face encodings
unique_persons = {}
person_count = 1

# Adjust the distance threshold (higher value makes it harder to classify as a new person)
distance_threshold = 0.7

# Face tracking variables
tracked_persons = {}
person_ids = {}
next_person_id = 1

face_recognition_skip_frames = 2

# Process the video
frame_count = 0
face_recognition_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    results = model(frame)

    # Draw bounding boxes and track faces
    annotated_frame = np.asarray(frame)
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Perform face recognition every face_recognition_skip_frames frames
        if face_recognition_counter % face_recognition_skip_frames == 0:
            fixed_x1 = max(center_x - fixed_screenshot_size[0] // 2, 0)
            fixed_y1 = max(center_y - fixed_screenshot_size[1] // 2, 0)
            fixed_x2 = min(fixed_x1 + fixed_screenshot_size[0], width)
            fixed_y2 = min(fixed_y1 + fixed_screenshot_size[1], height)

            cropped_face = frame[fixed_y1:fixed_y2, fixed_x1:fixed_x2]
            cropped_face = cv2.resize(cropped_face, fixed_screenshot_size)

            face_locations = face_recognition.face_locations(cropped_face)
            if face_locations:
                face_encoding = face_recognition.face_encodings(cropped_face, face_locations)[0]
                distances = {person: face_recognition.face_distance([embeddings], face_encoding)[0] for person, embeddings in unique_persons.items()}
                if distances:
                    min_distance = min(distances.values())
                    if min_distance < distance_threshold:
                        person_name = min(distances, key=distances.get)
                        person_id = person_ids[person_name]
                    else:
                        person_name = f"person_{person_count}"
                        unique_persons[person_name] = face_encoding
                        person_ids[person_name] = next_person_id
                        person_id = next_person_id
                        next_person_id += 1
                        person_count += 1
                else:
                    person_name = f"person_{person_count}"
                    unique_persons[person_name] = face_encoding
                    person_ids[person_name] = next_person_id
                    person_id = next_person_id
                    next_person_id += 1
                    person_count += 1

                # Save face screenshots with timestamp and person ID
                video_position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                filename_timestamp = datetime.fromtimestamp(video_position_ms / 1000).strftime('%M_%S')
                person_dir = os.path.join(base_dir, person_name)
                os.makedirs(person_dir, exist_ok=True)

                # Create a black space at the bottom of the image for the timestamp and person ID
                cropped_face_with_timestamp = np.zeros((fixed_screenshot_size[1] + timestamp_height * 2, fixed_screenshot_size[0], 3), dtype=np.uint8)
                cropped_face_with_timestamp[:fixed_screenshot_size[1], :] = cropped_face

                # Add timestamp and person ID to the black space
                screenshot_timestamp = datetime.fromtimestamp(video_position_ms / 1000).strftime('%M:%S')
                cv2.putText(cropped_face_with_timestamp, screenshot_timestamp, (10, fixed_screenshot_size[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(cropped_face_with_timestamp, f"ID: {person_id}", (10, fixed_screenshot_size[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                face_path = os.path.join(person_dir, f'face_{filename_timestamp}.jpg')
                cv2.imwrite(face_path, cropped_face_with_timestamp)

            else:
                print("Face detection failed for this frame.")

        # Face tracking
        if face_locations:
            if person_id in tracked_persons:
                tracked_persons[person_id].update([center_x, center_y])
            else:
                tracked_persons[person_id] = TrackingObject([center_x, center_y])
        else:
            # If face detection failed, skip tracking for this frame
            pass

        # Draw bounding box only
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add timestamp to the frame
    video_position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    timestamp = datetime.fromtimestamp(video_position_ms / 1000).strftime('%M:%S')
    cv2.putText(annotated_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(annotated_frame)

    frame_count += 1
    face_recognition_counter += 1
    print(f'Processed frame {frame_count}')
    cv2.imshow('Face Detection', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()