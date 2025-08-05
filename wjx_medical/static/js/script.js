// 全局变量
let currentFilename = null;
let masksInfo = [];
let currentTab = "segmentation";

// DOM元素
const uploadBtn = document.getElementById('uploadBtn');
const uploadArea = document.getElementById('uploadArea');
const imageFileInput = document.getElementById('imageFile');
const originalImage = document.getElementById('originalImage');
const denoisedImage = document.getElementById('denoisedImage');
const segmentationImage = document.getElementById('segmentationImage');
const measurementImage = document.getElementById('measurementImage');
const applyDenoiseBtn = document.getElementById('applyDenoiseBtn');
const segmentBtn = document.getElementById('segmentBtn');
const measureBtn = document.getElementById('measureBtn');
const denoiseMethod = document.getElementById('denoiseMethod');
const kernelSize = document.getElementById('kernelSize');
const kernelValue = document.getElementById('kernelValue');
const sigmaSlider = document.getElementById('sigmaSlider');
const sigmaValue = document.getElementById('sigmaValue');
const pixelsPerMm = document.getElementById('pixelsPerMm');
const maskSelector = document.getElementById('maskSelector');
const maskSelectorContainer = document.getElementById('maskSelectorContainer');
const widthValue = document.getElementById('widthValue');
const heightValue = document.getElementById('heightValue');
const areaValue = document.getElementById('areaValue');
const maskInfo = document.getElementById('maskInfo');
const deviceInfo = document.getElementById('deviceInfo');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingMessage = document.getElementById('loadingMessage');
const progressBar = document.getElementById('progressBar');
const themeToggle = document.getElementById('themeToggle');
const tabButtons = document.querySelectorAll('.tab-btn');

// 事件监听器
document.addEventListener('DOMContentLoaded', initApp);
uploadBtn.addEventListener('click', uploadImage);
uploadArea.addEventListener('click', () => imageFileInput.click());
imageFileInput.addEventListener('change', handleFileSelect);
applyDenoiseBtn.addEventListener('click', applyDenoising);
segmentBtn.addEventListener('click', segmentImage);
measureBtn.addEventListener('click', measureObject);
kernelSize.addEventListener('input', updateKernelValue);
sigmaSlider.addEventListener('input', updateSigmaValue);
maskSelector.addEventListener('change', selectMask);
themeToggle.addEventListener('click', toggleTheme);
tabButtons.forEach(btn => btn.addEventListener('click', switchTab));

// 初始化应用
function initApp() {
    updateKernelValue();
    updateSigmaValue();
    
    // 显示设备信息
    const isCuda = deviceInfo.textContent.includes('cuda');
    deviceInfo.textContent = isCuda ? 'GPU加速' : 'CPU处理';
    deviceInfo.className = isCuda ? 'text-success' : 'text-warning';
    
    // 设置初始标签页
    switchTab({target: document.querySelector('.tab-btn.active')});
}

// 处理文件选择
function handleFileSelect(e) {
    if (e.target.files.length) {
        uploadImage();
    }
}

// 更新核大小值
function updateKernelValue() {
    kernelValue.textContent = kernelSize.value;
}

// 更新Sigma值
function updateSigmaValue() {
    sigmaValue.textContent = (sigmaSlider.value / 10).toFixed(1);
}

// 上传图像
function uploadImage() {
    if (!imageFileInput.files.length) {
        showToast('请选择图像文件', 'warning');
        return;
    }
    
    showLoading('上传图像中...');
    
    const formData = new FormData();
    formData.append('file', imageFileInput.files[0]);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            hideLoading();
            showToast(data.error, 'error');
            return;
        }
        
        currentFilename = data.filename;
        originalImage.src = data.original;
        denoisedImage.src = 'https://via.placeholder.com/400x300?text=应用降噪后显示';
        segmentationImage.src = 'https://via.placeholder.com/600x400?text=分割结果将显示在这里';
        measurementImage.src = 'https://via.placeholder.com/600x400?text=测量结果将显示在这里';
        
        // 重置UI状态
        maskSelectorContainer.style.display = 'none';
        maskInfo.innerHTML = '';
        
        hideLoading();
        showToast('图像上传成功！', 'success');
    })
    .catch(error => {
        hideLoading();
        console.error('上传错误:', error);
        showToast('上传失败: ' + error.message, 'error');
    });
}

// 应用降噪
function applyDenoising() {
    if (!currentFilename) {
        showToast('请先上传图像', 'warning');
        return;
    }
    
    showLoading('降噪处理中...');
    updateProgress(30);
    
    const method = denoiseMethod.value;
    const kernel = kernelSize.value;
    const sigma = (sigmaSlider.value / 10).toFixed(1);
    
    fetch('/denoise', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: currentFilename,
            method: method,
            kernel_size: kernel,
            sigma: sigma
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            hideLoading();
            showToast(data.error, 'error');
            return;
        }
        
        denoisedImage.src = data.denoised;
        currentFilename = data.denoised_filename;
        updateProgress(100);
        setTimeout(() => {
            hideLoading();
            showToast('降噪处理完成！', 'success');
        }, 500);
    })
    .catch(error => {
        hideLoading();
        console.error('降噪错误:', error);
        showToast('降噪失败: ' + error.message, 'error');
    });
}

// 分割图像
function segmentImage() {
    if (!currentFilename) {
        showToast('请先上传图像', 'warning');
        return;
    }
    
    showLoading('图像分割中，请稍候...');
    updateProgress(20);
    
    fetch('/segment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: currentFilename
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            hideLoading();
            showToast(data.error, 'error');
            return;
        }
        
        segmentationImage.src = data.segmentation;
        masksInfo = data.masks_info;
        
        // 更新掩码选择器
        updateMaskSelector();
        
        // 显示区域信息
        displayMaskInfo();
        
        updateProgress(100);
        setTimeout(() => {
            hideLoading();
            showToast(`分割完成，找到 ${data.num_masks} 个区域`, 'success');
        }, 500);
    })
    .catch(error => {
        hideLoading();
        console.error('分割错误:', error);
        showToast('分割失败: ' + error.message, 'error');
    });
}

// 更新掩码选择器
function updateMaskSelector() {
    maskSelector.innerHTML = '';
    
    masksInfo.forEach((mask, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `区域 ${index + 1} (面积: ${mask.area})`;
        maskSelector.appendChild(option);
    });
    
    maskSelectorContainer.style.display = 'block';
}

// 显示掩码信息
function displayMaskInfo() {
    maskInfo.innerHTML = '<h4>区域信息:</h4><div class="mask-grid">';
    
    masksInfo.slice(0, 5).forEach((mask, index) => {
        maskInfo.innerHTML += `
            <div class="mask-card">
                <div class="mask-header">区域 ${index + 1}</div>
                <div class="mask-body">
                    <div><i class="fas fa-vector-square"></i> 面积: ${mask.area} 像素</div>
                    <div><i class="fas fa-bullseye"></i> 置信度: ${mask.predicted_iou.toFixed(3)}</div>
                    <div><i class="fas fa-border-all"></i> 边界框: ${mask.bbox.join(', ')}</div>
                </div>
            </div>
        `;
    });
    
    maskInfo.innerHTML += '</div>';
}

// 选择掩码
function selectMask() {
    // 可以在这里添加额外的处理逻辑
}

// 测量对象
function measureObject() {
    if (!currentFilename) {
        showToast('请先上传图像', 'warning');
        return;
    }
    
    if (!maskSelector.value && maskSelectorContainer.style.display === 'block') {
        showToast('请选择要测量的区域', 'warning');
        return;
    }
    
    showLoading('测量中...');
    updateProgress(40);
    
    const maskId = maskSelector.value || 0;
    const ppmm = pixelsPerMm.value;
    
    fetch('/measure', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: currentFilename,
            mask_id: maskId,
            pixels_per_mm: ppmm
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            hideLoading();
            showToast(data.error, 'error');
            return;
        }
        
        measurementImage.src = data.measurement;
        widthValue.textContent = data.width;
        heightValue.textContent = data.height;
        areaValue.textContent = data.area;
        
        // 更新图表
        updateChart(data.width, data.height, data.area);
        
        updateProgress(100);
        setTimeout(() => {
            hideLoading();
            showToast('测量完成！', 'success');
            switchTab({target: document.querySelector('[data-tab="measurement"]')});
        }, 500);
    })
    .catch(error => {
        hideLoading();
        console.error('测量错误:', error);
        showToast('测量失败: ' + error.message, 'error');
    });
}

// 更新图表
function updateChart(width, height, area) {
    const ctx = document.getElementById('measurementChart').getContext('2d');
    
    // 销毁现有图表实例（如果存在）
    if (window.measurementChart) {
        window.measurementChart.destroy();
    }
    
    window.measurementChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['宽度', '高度', '面积'],
            datasets: [{
                label: '测量结果 (mm)',
                data: [parseFloat(width), parseFloat(height), parseFloat(area)],
                backgroundColor: [
                    'rgba(67, 97, 238, 0.7)',
                    'rgba(76, 201, 240, 0.7)',
                    'rgba(247, 37, 133, 0.7)'
                ],
                borderColor: [
                    'rgba(67, 97, 238, 1)',
                    'rgba(76, 201, 240, 1)',
                    'rgba(247, 37, 133, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} mm${context.dataIndex === 2 ? '²' : ''}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '尺寸 (mm)',
                        color: '#a0a0c0'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0a0c0'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0a0c0'
                    }
                }
            }
        }
    });
}

// 显示加载指示器
function showLoading(message) {
    loadingMessage.textContent = message;
    loadingOverlay.style.display = 'flex';
    updateProgress(10);
}

// 更新进度条
function updateProgress(value) {
    progressBar.style.width = `${value}%`;
}

// 隐藏加载指示器
function hideLoading() {
    setTimeout(() => {
        loadingOverlay.style.display = 'none';
        progressBar.style.width = '0%';
    }, 300);
}

// 切换主题
function toggleTheme() {
    document.body.classList.toggle('light-theme');
    const icon = themeToggle.querySelector('i');
    if (document.body.classList.contains('light-theme')) {
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    } else {
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
    }
}

// 切换标签页
function switchTab(e) {
    const tab = e.target.getAttribute('data-tab');
    
    // 更新按钮状态
    tabButtons.forEach(btn => btn.classList.remove('active'));
    e.target.classList.add('active');
    
    // 更新内容
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tab}Tab`).classList.add('active');
    
    currentTab = tab;
}

// 显示Toast通知
function showToast(message, type) {
    // 创建Toast元素
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-icon">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
        </div>
        <div class="toast-message">${message}</div>
        <div class="toast-close"><i class="fas fa-times"></i></div>
    `;
    
    // 添加到页面
    document.body.appendChild(toast);
    
    // 显示Toast
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    // 自动隐藏
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 5000);
    
    // 添加关闭事件
    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    });
}