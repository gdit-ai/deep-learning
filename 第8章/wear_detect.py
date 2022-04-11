from predict_wear.yolo import YOLO
from PIL import Image
import numpy as np


def define_yolo():
        import argparse
        # class YOLO defines the default value, so suppress any default here
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        '''
        Command line options
        '''
        parser.add_argument(
            '--model', type=str,
            help='path to model weight file, default ' + YOLO.get_defaults("model_path")
        )

        parser.add_argument(
            '--anchors', type=str,
            help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
        )

        parser.add_argument(
            '--classes', type=str,
            help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
        )

        parser.add_argument(
            '--gpu_num', type=int,
            help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
        )

        parser.add_argument(
            '--image', default=False, action="store_true",
            help='Image detection mode, will ignore all positional arguments'
        )
        '''
        Command line positional arguments -- for video detection mode
        '''
        parser.add_argument(
            "--input", nargs='?', type=str, required=False, default='./path2your_video',
            help="Video input path"
        )

        parser.add_argument(
            "--output", nargs='?', type=str, default="",
            help="[Optional] Video output path"
        )

        FLAGS = parser.parse_args()
        FLAGS.model_path = "./predict_wear/logs/trained_weights_final.h5"
        FLAGS.anchors_path = "./predict_wear/model_data/yolo_anchors.txt"
        # FLAGS.anchors = "./predict_wear/model_data/yolo_anchors.txt"
        # FLAGS.classes = "./predict_wear/model_data/coco_classes.txt"
        # FLAGS.classes_path = "./model_data/coco_classes.txt"
        FLAGS.classes_path = "./predict_wear/model_data/voc_classes.txt"

        m_yolo = YOLO(**vars(FLAGS))
        return m_yolo

def wear_detect_call():  # 展示人脸检测的功能
    import cv2
    cap = cv2.VideoCapture("class_wear.mp4")
    num = 0
    m_yolo = define_yolo()
    curr_fps = 0
    while True:
        return_value, frame = cap.read()
        image = cv2.resize(frame, (960, 540))
        image = Image.fromarray(image)
        image, detect_object_num, detect_object_list, detect_object_box = m_yolo.detect_image(image)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    m_yolo.close_session()

wear_detect_call()