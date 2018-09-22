import face_recognition
import numpy as np
from multiprocessing import Process
import cv2
import Tkinter

def npz_load (file):
    encodings = np.load(file) 
    c = []
    for i in encodings.files:
        for arr in encodings[i]:
            c.append(arr);
    return c

def npz_dumps (file, encs):
    np.savez(file, encs)

def name_dumps(file, names):
    w_file_names = open(f_names, 'w')
    for name in names:
        w_file_names.write(name + "\n")

def name_load (file):
    r_file_names = open(f_names, 'r')
    all_lines = r_file_names.readlines()
    known_names = []
    for line in all_lines:
        known_names.append(line)
    return known_names

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
f_name_enc = "face_enc.npz"
f_names = "face_names.txt"

r_file_encodings = open(f_name_enc, 'r')
r_file_names = open(f_names, 'r')

a_file_encodings = open(f_name_enc, 'a')
# Load a sample picture and learn how to recognize it.
my_image = face_recognition.load_image_file("./imgs/davi.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]

# Create arrays of known face encodings and their names
file_restart = False
if file_restart:
    known_face_encodings = []
    known_face_names = []
else:
    known_face_encodings = npz_load(f_name_enc)
    known_face_names = name_load(f_names)

all_encodings = r_file_encodings.readline()
print all_encodings
all_names = r_file_encodings.readlines()
#all_names = names.split(",")
#all_encodings = encodings.split(",")

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
unknown_face_encodings = []

# a_file_encodings.write(np.array_str(a)+"\n")

# all_lines = r_file_encodings.readlines()
# with_brackets = []
# # print all_lines
# i = 0
# for line in all_lines:
#     with_brackets = line.split("\n");
#     print "ola"
#     print with_brackets
#     last_element = with_brackets[len(with_brackets)-1]
#     with_brackets[len(with_brackets)-1] = last_element[:-2]
#     with_brackets[0] = with_brackets[0][1:]
#     print with_brackets[0]
#     all_lines[i] = with_brackets
#     i = i + 1
# for line in all_lines:
#     print line

# i = 0
# for line in all_lines:
#     array_encodings = np.array([])
#     for i in range(len(line)):
#         numero = str.isdigit(line[i])
#         if numero:
#             print line[i]
#             array_encodings = np.append(array_encodings,int(line[i]))
#             i = i + 1
#     face_encodings.append(array_encodings)

# all_lines = r_file_names.readlines()
# for name in all_lines:
#     face_names.append(name)

count = 0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # if count < 10:
    #     process_this_frame = False
    #     count += 1
    #     print count
    #     print process_this_frame
    # elif count == 10:
    #     process_this_frame = True

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            if name == "unknown":
                unknown_face_encodings.append(face_encoding)
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results   
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4    
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    if len(unknown_face_encodings)!=0:
        for(face_encoding) in (unknown_face_encodings):
            new_name = raw_input("Whats your name? ")
            known_face_encodings.append(face_encoding)
            known_face_names.append(new_name)
            # a_file_encodings.write(np.array_str(face_encoding)+"\n")
            # a_file_names.write(new_name + "\n")
        unknown_face_encodings = []
    else:
        # Display the resulting image
        big_frame = cv2.resize(frame, (0, 0), fx=4, fy=4)
        cv2.imshow('Video', big_frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release handle to the webcam
npz_dumps(f_name_enc, known_face_encodings)
name_dumps(f_names, known_face_names)

video_capture.release()
cv2.destroyAllWindows()