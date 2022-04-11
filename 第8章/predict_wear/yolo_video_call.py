import sys
import argparse
# from yolo import YOLO, detect_video
from predict_wear.yolo import YOLO, detect_video

def detect_img(yolo, image):
    r_image = yolo.detect_image(image)
    # r_image.show()
    yolo.close_session()

FLAGS = None

def predict_wear(image):
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
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    #hujianhua
    # --model=model_data/ep024-loss0.158-val_loss0.144.h5
    # --anchors=model_data/yolo_anchors.txt
    # --classes=model_data/coco_classes.txt
    # --image
    FLAGS.image = True

    # FLAGS.model = "./predict_wear/model_data/ep024-loss0.158-val_loss0.144.h5"
    # FLAGS.model_path = "./predict_wear/model_data/yolo.h5"
    # FLAGS.model_path = "./logs/ep024-loss0.158-val_loss0.144.h5"
    # FLAGS.model_path = "./logs/trained_weights_stage_1.h5"
    FLAGS.model_path = "./logs/trained_weights_final.h5"
    FLAGS.anchors_path = "./model_data/yolo_anchors.txt"
    # FLAGS.anchors = "./predict_wear/model_data/yolo_anchors.txt"
    # FLAGS.classes = "./predict_wear/model_data/coco_classes.txt"
    # FLAGS.classes_path = "./model_data/coco_classes.txt"
    FLAGS.classes_path = "./model_data/voc_classes.txt"

    detect_img(YOLO(**vars(FLAGS)), image)


# img_path = "./img/42.jpg"
# predict_wear(img_path)