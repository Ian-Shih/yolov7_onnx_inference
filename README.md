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

### run onnx_inference.py
<pre><code>cd ..
python onnx_inference.py --input-image input\walk.jpg --model-path yolov7\yolov7.onnx --output-folder output --conf-thres 0.6 --iou-thres 0.4  
</code></pre>

### output image
![Alt text](/output/walk.jpg)

### output txt
<pre><code>x1	y1	x2	y2	bbox_conf	class_conf	class
417.089111328125	82.91847229003906	468.669921875	343.479248046875	0.8768256902694702	0.8767907023429871	0.0
318.99261474609375	17.335662841796875	404.59698486328125	391.7721862792969	0.8746476769447327	0.8746188282966614	0.0
230.85130310058594	82.89126586914062	287.5687255859375	343.74005126953125	0.8384268283843994	0.8384206295013428	0.0
91.86311340332031	83.73114013671875	162.22314453125	358.01708984375	0.7793647050857544	0.7793200612068176	0.0
0.28083038330078125	123.6357192993164	91.28294372558594	253.46365356445312	0.792332649230957	0.7894709706306458	2.0
107.35469055175781	112.89877319335938	237.63841247558594	236.79415893554688	0.6636846661567688	0.6564241647720337	2.0
0.2469940185546875	254.04547119140625	46.7682991027832	385.3057861328125	0.797244668006897	0.7966926097869873	3.0
111.32697296142578	126.38291931152344	154.28138732910156	241.2587127685547	0.7211576700210571	0.7186387777328491	26.0
</code></pre>

### cpu inference time
<pre><code>Process Time: 0.561929 秒
</code></pre>
