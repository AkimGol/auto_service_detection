import cv2
from nomeroff_net import pipeline

number_plate_short_detection_and_reading = pipeline("number_plate_short_detection_and_reading", 
                                                    text_reader_name="by") # image_loader="opencv"

path = 'video/1.mp4'
cap = cv2.VideoCapture(path)

while(True): 
    ret, frame = cap.read()
    #print(number_plate_short_detection_and_reading([frame]))
    # frame = cv2.rectangle(frame, (384, 0), (510, 128), (0, 255, 0), 10) 
    img, img_box, zone, text = number_plate_short_detection_and_reading([frame])[0]
    if len(img_box) != 0:
        x0, y0, x1, y1, conf, _ = img_box[0]
        if conf > 0.6:
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.putText(frame, text[0], (int(x0), int(y1)+30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    k = 0.6
    cv2.imshow('car', cv2.resize(frame, (int(1920*k), int(1080*k))))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
