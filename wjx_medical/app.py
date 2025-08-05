import os
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from werkzeug.utils import secure_filename
from skimage import measure
import base64
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
app.config['MODEL_PATH'] = 'models/sam_vit_h_4b8939.pth'

# 确保上传目录存在 - 修复目录创建问题
def ensure_upload_folder():
    upload_folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_folder):
        # 如果路径存在但不是一个目录，删除它
        if not os.path.isdir(upload_folder):
            os.remove(upload_folder)
            os.makedirs(upload_folder)
    else:
        os.makedirs(upload_folder)

# 在应用启动时创建上传目录
ensure_upload_folder()

# 全局变量
sam_model = None
mask_generator = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_sam_model():
    """加载SAM模型"""
    global sam_model, mask_generator
    
    if sam_model is None:
        print("正在加载SAM模型...")
        start_time = time.time()
        
        # 检查模型文件是否存在
        if not os.path.exists(app.config['MODEL_PATH']):
            raise FileNotFoundError(f"模型文件 {app.config['MODEL_PATH']} 未找到")
        
        # 加载模型
        model_type = "vit_h"
        sam_model = sam_model_registry[model_type](checkpoint=app.config['MODEL_PATH'])
        sam_model.to(device=device)
        
        # 创建掩码生成器
        mask_generator = SamAutomaticMaskGenerator(
            sam_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )
        
        print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")
    
    return mask_generator

def apply_denoising(image, method, kernel_size, sigma=1.5):
    """应用降噪处理"""
    if method == "gaussian":
        # 确保核大小为奇数
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    elif method == "mean":
        return cv2.blur(image, (kernel_size, kernel_size))
    elif method == "median":
        # 确保核大小为奇数
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.medianBlur(image, kernel_size)
    return image

def process_image_to_base64(image):
    """将处理后的图像转换为Base64格式"""
    _, buffer = cv2.imencode('.jpg', image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

def process_image_to_matplotlib(image):
    """将处理后的图像转换为Matplotlib Base64格式"""
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(image.shape) == 2:  # 灰度图
        ax.imshow(image, cmap='gray')
    else:  # 彩色图
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
    ax.axis('off')
    
    # 将图形保存到缓冲区
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

def clear_upload_folder():
    """清空上传文件夹"""
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'删除 {file_path} 失败. 原因: {e}')

@app.route('/')
def index():
    """主页面"""
    # 清空上传文件夹
    clear_upload_folder()
    return render_template('index.html', device=device)

@app.route('/upload', methods=['POST'])
def upload_image():
    """上传图像"""
    if 'file' not in request.files:
        return jsonify({"error": "没有文件部分"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 读取图像
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({"error": "无法读取图像文件"}), 400
        
        # 返回原始图像和尺寸信息
        height, width = image.shape[:2]
        return jsonify({
            "filename": filename,
            "original": process_image_to_base64(image),
            "width": width,
            "height": height
        })
    
    return jsonify({"error": "未知错误"}), 500

@app.route('/denoise', methods=['POST'])
def denoise_image():
    """应用降噪处理"""
    data = request.json
    filename = data.get('filename')
    method = data.get('method', 'gaussian')
    kernel_size = int(data.get('kernel_size', 5))
    sigma = float(data.get('sigma', 1.5))
    
    if not filename:
        return jsonify({"error": "缺少文件名"}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "无法读取图像文件"}), 400
    
    # 应用降噪
    denoised = apply_denoising(image, method, kernel_size, sigma)
    
    # 保存降噪后的图像
    denoised_filename = f"denoised_{filename}"
    denoised_path = os.path.join(app.config['UPLOAD_FOLDER'], denoised_filename)
    cv2.imwrite(denoised_path, denoised)
    
    # 返回结果
    return jsonify({
        "denoised": process_image_to_base64(denoised),
        "denoised_filename": denoised_filename
    })

@app.route('/segment', methods=['POST'])
def segment_image():
    """分割图像"""
    data = request.json
    filename = data.get('filename')
    
    if not filename:
        return jsonify({"error": "缺少文件名"}), 400
    
    # 加载模型
    try:
        mask_generator = load_sam_model()
    except Exception as e:
        return jsonify({"error": f"加载SAM模型失败: {str(e)}"}), 500
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "无法读取图像文件"}), 400
    
    # 分割图像
    start_time = time.time()
    try:
        masks = mask_generator.generate(image)
        print(f"分割完成，找到 {len(masks)} 个区域，耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        return jsonify({"error": f"分割失败: {str(e)}"}), 500
    
    # 创建分割结果可视化
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if len(masks) > 0:
        # 按面积排序
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        
        # 只显示前10个最大的区域
        for i, mask in enumerate(sorted_masks[:10]):
            segmentation = mask['segmentation']
            color = np.random.random(3)
            img = np.ones((segmentation.shape[0], segmentation.shape[1], 3))
            for j in range(3):
                img[:, :, j] = color[j]
            ax.imshow(np.dstack((img, segmentation * 0.35)))
    
    ax.axis('off')
    
    # 将图形保存到缓冲区
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    segmentation_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
    
    # 保存掩码信息
    masks_info = []
    for i, mask in enumerate(masks[:10]):  # 只返回前10个区域的信息
        masks_info.append({
            "id": i,
            "area": mask['area'],
            "bbox": mask['bbox'],
            "predicted_iou": mask['predicted_iou']
        })
    
    return jsonify({
        "segmentation": segmentation_base64,
        "num_masks": len(masks),
        "masks_info": masks_info
    })

@app.route('/measure', methods=['POST'])
def measure_object():
    """测量对象尺寸"""
    data = request.json
    filename = data.get('filename')
    mask_id = int(data.get('mask_id', 0))
    pixels_per_mm = float(data.get('pixels_per_mm', 10.0))
    
    if not filename:
        return jsonify({"error": "缺少文件名"}), 400
    
    # 加载模型
    try:
        mask_generator = load_sam_model()
    except Exception as e:
        return jsonify({"error": f"加载SAM模型失败: {str(e)}"}), 500
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "无法读取图像文件"}), 400
    
    # 分割图像
    try:
        masks = mask_generator.generate(image)
    except Exception as e:
        return jsonify({"error": f"分割失败: {str(e)}"}), 500
    
    if not masks:
        return jsonify({"error": "未找到分割区域"}), 400
    
    # 确保mask_id在有效范围内
    if mask_id >= len(masks):
        mask_id = len(masks) - 1
    
    # 获取指定掩码
    mask = masks[mask_id]
    segmentation = mask['segmentation']
    
    # 创建测量可视化图像
    image_vis = image.copy()
    
    # 找到轮廓
    try:
        contours = measure.find_contours(segmentation.astype(np.uint8), 0.5)
    except Exception as e:
        return jsonify({"error": f"轮廓检测失败: {str(e)}"}), 500
    
    if contours:
        # 取最大轮廓
        contour = max(contours, key=len)
        
        # 计算最小外接矩形
        try:
            rect = cv2.minAreaRect(np.array(contour, dtype=np.int32))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
        except Exception as e:
            return jsonify({"error": f"计算外接矩形失败: {str(e)}"}), 500
        
        # 在图像上绘制矩形
        cv2.drawContours(image_vis, [box], 0, (0, 255, 0), 2)
        
        # 计算尺寸（毫米）
        width = max(rect[1][0], rect[1][1]) / pixels_per_mm
        height = min(rect[1][0], rect[1][1]) / pixels_per_mm
        area = mask['area'] / (pixels_per_mm ** 2)
        
        # 添加尺寸标签
        text = f"尺寸: {width:.2f} x {height:.2f} mm"
        cv2.putText(image_vis, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加面积信息
        area_text = f"面积: {area:.2f} mm²"
        cv2.putText(image_vis, area_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 返回测量结果
        return jsonify({
            "measurement": process_image_to_base64(image_vis),
            "width": f"{width:.2f}",
            "height": f"{height:.2f}",
            "area": f"{area:.2f}"
        })
    
    return jsonify({"error": "无法测量对象"}), 400

if __name__ == '__main__':
    # 清空上传文件夹
    clear_upload_folder()
    # 确保上传目录存在
    ensure_upload_folder()
    app.run(host='0.0.0.0', port=5000, debug=True)