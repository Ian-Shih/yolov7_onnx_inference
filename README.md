# yolov7_onnx_inference

### cd to "yolov7/" and run these commands

<pre><code># install yolov7 requirements
pip install -r requirements.txt

# install requirements for torch gpu
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# install requirements for export onnx
pip install onnx==1.14.1 onnxruntime==1.14.1
  
# export onnx
python export.py --weights yolov7.pt --grid --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640 </code></pre>

### edit onnx_inference.py and run
<pre><code># set input,output,and model path
	
in_image_path='..\input\ohtani.jpg'
out_folder_path='..\output'
onnx_model_path = 'yolov7.onnx'
</code></pre>

<pre><code># customize iou threshold and class threshold
	
result=non_max_suppression_manual(np.array(outputs[0]), conf_thres=0.65, iou_thres=0.35)
</code></pre>

### output image
![Alt text](/output/dog.jpg)

### cpu inference time
<pre><code>Process Time: 0.561929 秒
</code></pre>
