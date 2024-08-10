import numpy as np
import cv2
def calculate_iou(boxes1, boxes2):
    x1_1, y1_1, x2_1, y2_1 = boxes1[0], boxes1[1], boxes1[2], boxes1[3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[0], boxes2[1], boxes2[2], boxes2[3]

    xx1 = np.maximum(x1_1, x1_2)
    yy1 = np.maximum(y1_1, y1_2)
    xx2 = np.minimum(x2_1, x2_2)
    yy2 = np.minimum(y2_1, y2_2)

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h

    areas1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    areas2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    iou = inter / (areas1 + areas2 - inter)
    return iou

def draw_boxes(image, detections, class_names):
    for det in detections:
        x1, y1, x2, y2, bbox_conf, cls_conf, cls = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        class_id = int(cls)

        label = f'{class_names[class_id]}: {cls_conf:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def convert_to_corners(boxes):
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return np.stack((x1, y1, x2, y2), axis=1)

def non_max_suppression_manual(predictions, conf_thres=0.65, iou_thres=0.35):

    boxes = predictions[..., :4]
    boxes=convert_to_corners(boxes) # very important

    bbox_conf = predictions[..., 4:5]
    class_conf = predictions[..., 5:]
    max_scores = np.max(class_conf, axis=-1, keepdims=True)
    max_scores = max_scores*bbox_conf # very important
    max_classes = np.expand_dims(np.argmax(class_conf, axis=-1), axis=-1)
    bbox_status = np.concatenate((boxes, bbox_conf, max_scores, max_classes), axis=-1)
    
    # confidence filter
    mask = bbox_status[...,5] >= conf_thres
    filtered_data = bbox_status[mask]

    # unique classes
    unique_cls=np.unique(filtered_data[...,6])
    selected_bbox=[]

    for cls in unique_cls:

        mask = filtered_data[...,6] == cls
        cls_data = filtered_data[mask]

        # dec sort by class_conf
        sort_indices = np.argsort(cls_data[...,5])[::-1]
        sorted_data = cls_data[sort_indices]

        while len(sorted_data) > 0:
            best_box = sorted_data[0]
            selected_bbox.append(best_box)

            # best vs other
            ious = np.array([calculate_iou(best_box[:4], box) for box in sorted_data[1:]])

            # iou filter
            remaining_indices = np.where(ious <= iou_thres)[0] + 1
            sorted_data = sorted_data[remaining_indices]
    
    return selected_bbox