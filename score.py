import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.layers import Input

from src.utils.fixes import fix_tf_gpu
from src.utils.image import draw_detection, letterbox_image
from src.yolo3.detect import detection
from src.yolo3.model import yolo_body


def prepare_model(approach):
    """
    Prepare the YOLO model
    """
    global input_shape, class_names, anchor_boxes, num_classes, num_anchors, model

    # shape (height, width) of the imput image
    input_shape = (416, 416)

    # class names
    if approach == 1:
        class_names = ["H", "V", "W"]

    elif approach == 2:
        class_names = ["W", "WH", "WV", "WHV"]

    elif approach == 3:
        class_names = ["W"]

    else:
        raise NotImplementedError("Approach should be 1, 2, or 3")

    # anchor boxes
    if approach == 1:
        anchor_boxes = np.array(
            [
                np.array([[76, 59], [84, 136], [188, 225]])
                / 32,  # output-1 anchor boxes
                np.array([[25, 15], [46, 29], [27, 56]]) / 16,  # output-2 anchor boxes
                np.array([[5, 3], [10, 8], [12, 26]]) / 8,  # output-3 anchor boxes
            ],
            dtype="float64",
        )
    else:
        anchor_boxes = np.array(
            [
                np.array([[73, 158], [128, 209], [224, 246]])
                / 32,  # output-1 anchor boxes
                np.array([[32, 50], [40, 104], [76, 73]]) / 16,  # output-2 anchor boxes
                np.array([[6, 11], [11, 23], [19, 36]]) / 8,  # output-3 anchor boxes
            ],
            dtype="float64",
        )

    # number of classes and number of anchors
    num_classes = len(class_names)
    num_anchors = anchor_boxes.shape[0] * anchor_boxes.shape[1]

    # input and output
    input_tensor = Input(shape=(input_shape[0], input_shape[1], 3))  # input
    num_out_filters = (num_anchors // 3) * (5 + num_classes)  # output

    # build the model
    model = yolo_body(input_tensor, num_out_filters)

    # load weights
    weight_path = Path(f"/artefact/pictor-ppe-v302-a{approach}-yolo-v3-weights.h5")
    if not weight_path.exists():
        weight_path = Path(f".{weight_path}")
    model.load_weights(str(weight_path))


def get_detection(img):
    # shape of the image
    ih, iw = img.shape[:2]

    # preprocess the image
    img = np.expand_dims(img, 0)
    image_data = np.array(img) / 255.0

    # raw prediction from yolo model
    prediction = model.predict(image_data)

    # process the raw prediction to get the bounding boxes
    boxes = detection(
        prediction,
        anchor_boxes,
        num_classes,
        image_shape=(ih, iw),
        input_shape=(416, 416),
        max_boxes=10,
        score_threshold=0.3,
        iou_threshold=0.45,
        classes_can_overlap=False,
    )

    # convert tensor to numpy
    return boxes[0].numpy()


def main(path):
    fix_tf_gpu()
    prepare_model(approach=2)
    cap = cv2.VideoCapture(path)
    # Create video encoder
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    output = Path("output") / Path(path).stem
    output.parent.mkdir(exist_ok=True)
    encoder = cv2.VideoWriter(f"{output}.mp4", codec, fps, (width, height))

    scale = min(input_shape[1] / width, input_shape[0] / height)
    offset_x = input_shape[1] - int(width * scale)
    offset_y = input_shape[0] - int(height * scale)

    result = ["frame,x1,y1,x2,y2,score,label\n"]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # save a copy of the img
        act_img = frame.copy()
        frame = letterbox_image(frame, input_shape)
        boxes = get_detection(frame)
        boxes[:, [0, 2]] -= offset_x // 2
        boxes[:, [1, 3]] -= offset_y // 2
        boxes[:, :4] /= scale
        # save per frame results
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        for bbox in boxes:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            # reclassify using cropped image
            cropped = act_img[
                max(y1 - 50, 0) : min(y2 + 50, height),
                max(x1 - 50, 0) : min(x2 + 50, width),
            ]
            cropped = letterbox_image(cropped, input_shape)
            detection = get_detection(cropped)
            if detection.size > 0:
                w = detection[:, 2] - detection[:, 0]
                h = detection[:, 3] - detection[:, 1]
                largest = np.argmax(w * h)
                bbox[-2] = detection[largest][-2]
                bbox[-1] = detection[largest][-1]
            # write to csv
            score = bbox[-2]
            label = int(bbox[-1])
            line = f"{pos},{x1},{y1},{x2},{y2},{score:.6f},{label}\n"
            result.append(line)
        # draw the detection on the actual image
        frame = draw_detection(act_img, boxes, class_names)
        # cv2.imwrite(f"{output}.png", frame)
        encoder.write(frame)
    cap.release()

    with open(f"{output}.txt", "w") as f:
        f.writelines(result)


if __name__ == "__main__":
    path = "extras/sample.mp4"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    main(path)
