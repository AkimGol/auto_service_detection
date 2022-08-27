from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
import os
import cv2


number_plate_detection_and_reading = pipeline("number_plate_short_detection_and_reading", 
                                                image_loader="opencv", text_reader_name="by")

# (images, images_bboxs,
#  zones, texts) = unzip(number_plate_detection_and_reading(['D:/sto_project/nomeroff-net/my_imgs/street/1687ie2.jpg']))
                                                        #    'D:/sto_project/nomeroff-net/my_imgs/street/1221ik2.jpg',
                                                        #    'D:/sto_project/nomeroff-net/my_imgs/street/2781ie2.jpg',
                                                        #    'D:/sto_project/nomeroff-net/my_imgs/street/5136bx2.jpg',
                                                        #    'D:/sto_project/nomeroff-net/my_imgs/street/5499ih2.jpg',
                                                        #    'D:/sto_project/nomeroff-net/my_imgs/street/6730ip2.jpg',
                                                        #    'D:/sto_project/nomeroff-net/my_imgs/street/7330bc2.jpg',
                                                        #    'D:/sto_project/nomeroff-net/my_imgs/street/9998ex2.jpg']))

dictionary = 'D:/sto_project/nomeroff-net/my_imgs/street'

for filename in os.listdir(dictionary):
    path = dictionary + '/' + filename
    img, img_box, zone, text = number_plate_detection_and_reading([path])[0]
    print(text)
    print(img_box)
    output = img.copy()
    x0, y0, x1, y1, _, _ = img_box[0]
    img_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img_rgb, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
    cv2.putText(img_rgb, text[0], (int(x0), int(y1)+30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    k = 0.6
    cv2.imshow('car', cv2.resize(img_rgb, (int(1920*k), int(1080*k))))
    cv2.waitKey(0)

