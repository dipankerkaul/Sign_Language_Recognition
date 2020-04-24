import cv2
import sys
import tensorflow as tf
import numpy as np
import statistics

def main():


    print("Do you want to open the webcam?")
    while True:

        guess = input('= ')
        if guess=="yes":
            video = cv2.VideoCapture(0)
            model = tf.keras.models.load_model(r'Weights/SignLanguageYesNoBadCNN2Epochs12.h5',compile=False)
            classes = {0: 'Bad', 1: 'No', 2: 'Yes'}
            bbox_initial = (50, 170, 150, 150)
            bbox = bbox_initial
            frame_count=0
            sentence = "shuruat"
            class_counter = []
            prevgesture = "bad"
            video = cv2.VideoCapture(0)







            while True:



                ok, frame = video.read()
                frame_count += 1

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hand_crop = 255 - gray[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                # Display the resulting frame

                try:

                    hand_crop_resized = np.expand_dims(cv2.resize(hand_crop, (150, 150)), axis=0).reshape(
                        (1, 150, 150, 1))
                    hand_crop_resized = np.repeat(hand_crop_resized, 3, -1)

                    prediction = model.predict(hand_crop_resized)

                    predi = prediction[0].argmax()  # Get the index of the greatest confidence

                    gesture = classes[predi]

                    if prevgesture != gesture:
                        #print(gesture)
                        prevgesture = gesture

                    if frame_count != 100:
                        class_counter.append(predi)
                    elif frame_count == 100:
                        max_class = statistics.mode(class_counter)
                        sentence = sentence +" "+ str(classes[max_class])
                        class_counter = []
                        frame_count = 0
                        print(sentence)
                    cv2.putText(frame, "p:{} s:{}".format(gesture, sentence), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(30, 0, 255), 2)

                except:
                    print("Shit has gone down!")
                cv2.imshow("frame", frame)
                cv2.imshow("hand_crop", hand_crop)

                ok, frame = video.read()

                cv2.imshow("frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
            video.release()










        elif guess=="exit":
            sys.exit()
        else:
            print("Type appropriate response please!")


main()