# coding: utf-8
import sys
import dlib
import cv2
import os

cur_path = os.getcwd()  # 当前目录路径
model_dir_path = cur_path + "/models/"  # models文件夹路径
face_dir_path = cur_path + "/faces/"  # faces文件夹路径

m = model_dir_path + sys.argv[1]

# 导入cnn模型才点
cnn_face_detector = dlib.cnn_face_detection_model_v1(m)

for f in sys.argv[2:]:
    # opencv 读取图片，并显示
    f1 = face_dir_path + f
    img = cv2.imread(f1, cv2.IMREAD_COLOR)

    # opencv的bgr格式图片转换成rgb格式
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])

    # 进行检测
    dets = cnn_face_detector(img, 1)

    # 打印检测到的人脸数
    print("Number of faces detected: {}".format(len(dets)))
    # 遍历返回的结果
    # 返回的结果是一个mmod_rectangles对象。这个对象包含有2个成员变量：dlib.rectangle类，表示对象的位置；dlib.confidence，表示置信度。
    for i, d in enumerate(dets):
        face = d.rect
        print(
            "Detect {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".
            format(i, face.left(), face.top(), face.right(), d.rect.bottom(),
                   d.confidence))

        # 在图片中标出人脸
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f, img)

k = cv2.waitKey(0)
cv2.destroyAllWindows()
