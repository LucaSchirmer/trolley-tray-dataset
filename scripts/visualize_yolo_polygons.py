import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

# Global variables to hold state
IMAGE_PATHS = []
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.JPG', '.JPEG')

def get_masked_image_base64(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    height, width, _ = img.shape
    name_part, _ = os.path.splitext(img_path)
    txt_path = f"{name_part}.txt"

    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            class_id = parts[0]
            coords = [float(x) for x in parts[1:]]
            
            pts = []
            for i in range(0, len(coords), 2):
                x_pixel = int(coords[i] * width)
                y_pixel = int(coords[i+1] * height)
                pts.append([x_pixel, y_pixel])
                
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            if len(pts) > 0:
                text_pos = (int(pts[0][0][0]), int(pts[0][0][1]) - 5)
                cv2.putText(img, f"Class {class_id}", text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(img, "No .txt file found", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Encode image to base64 so it can be rendered natively in HTML
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #222; color: #fff; padding: 20px; }
        img { max-width: 80%; max-height: 70vh; border: 3px solid #444; margin-top: 10px; }
        .btn { padding: 10px 20px; font-size: 16px; margin: 10px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; }
        .btn:hover { background: #0056b3; }
        .info { font-size: 18px; margin-bottom: 10px; color: #aaa; }
    </style>
</head>
<body>
    <h2>YOLO Dataset Viewer</h2>
    <div class="info">Image {{ index + 1 }} of {{ total }} — {{ filename }}</div>
    <div>
        <img src="data:image/jpeg;base64,{{ img_data }}" />
    </div>
    <div>
        {% if index > 0 %}
            <a href="/view/{{ index - 1 }}"><button class="btn">Previous</button></a>
        {% endif %}
        {% if index < total - 1 %}
            <a href="/view/{{ index + 1 }}"><button class="btn">Next</button></a>
        {% endif %}
    </div>
    <p style="color: #666;">Close your terminal window or press Ctrl+C to exit.</p>
</body>
</html>
"""

@app.route('/')
def index():
    if not IMAGE_PATHS:
        return "No images loaded. Make sure the input path is correct."
    return redirect(url_for('view_image', idx=0))

@app.route('/view/<int:idx>')
def view_image(idx):
    if idx < 0 or idx >= len(IMAGE_PATHS):
        return redirect(url_for('index'))
    
    img_path = IMAGE_PATHS[idx]
    img_data = get_masked_image_base64(img_path)
    filename = os.path.basename(img_path)
    
    return render_template_string(
        HTML_TEMPLATE, 
        img_data=img_data, 
        index=idx, 
        total=len(IMAGE_PATHS), 
        filename=filename
    )

def start_viewer(target_input):
    global IMAGE_PATHS
    if isinstance(target_input, str) and os.path.isdir(target_input):
        for f in os.listdir(target_input):
            if f.endswith(VALID_EXTENSIONS):
                IMAGE_PATHS.append(os.path.join(target_input, f))
    elif isinstance(target_input, list):
        IMAGE_PATHS = [p for p in target_input if os.path.exists(p) and p.endswith(VALID_EXTENSIONS)]
        
    if not IMAGE_PATHS:
        print("No valid JPEG images found.")
        return

    print("\n Web server starting! Copy and paste this URL into your Windows browser:")
    print(" --> http://127.0.0.1:5000\n")
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == '__main__':
    # Pass your directory here
    start_viewer('./delete_me_do_not_push')