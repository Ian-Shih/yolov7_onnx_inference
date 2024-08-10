# Inference for ONNX model
import onnxruntime as ort
import cv2
import numpy as np
from nms import non_max_suppression_manual,draw_boxes
import time
import os

# 80 class
names=[ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',\
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',\
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',\
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',\
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',\
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',\
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',\
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',\
         'hair drier', 'toothbrush' ]

# gpu
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
print(ort.get_device())


in_image_path='input\ohtani.jpg'
out_folder_path='output'
onnx_model_path = 'yolov7\yolov7.onnx'

start_time = time.time()

# image preprocess
img = cv2.imread(in_image_path)
img = cv2.resize(img,(640,640))
image = img.copy()
transpose_image = image.transpose((2, 0, 1))
transpose_image = np.expand_dims(transpose_image, 0)
inputs = transpose_image.astype(np.float32)
inputs /= 255

session = ort.InferenceSession(onnx_model_path,providers=providers)

# get onnx model input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# inference
outputs = session.run([output_name], {input_name: inputs})[0]

# nms
result=non_max_suppression_manual(np.array(outputs[0]))

# draw bboxes
result_image=draw_boxes(img,result,names)

# save result
os.makedirs(out_folder_path, exist_ok=True)
cv2.imwrite(os.path.join(out_folder_path, os.path.basename(in_image_path)),result_image)

# inference time
end_time = time.time()
print("Process Time: {:.6f} 秒".format(end_time - start_time))

