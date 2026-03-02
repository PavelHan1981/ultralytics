import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. 加载分割模型
model = YOLO('_my_research/pre_trained/yolov8l-seg.pt') 

def process_and_save_video():
    # 获取当前项目目录的绝对路径
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = "D:/Datasets/traffic_video/vecteezy_aerial-view-of-british-roads-and-traffic-passing-through_18137910.mp4" # 替换为你的视频路径
    output_path = os.path.join(project_dir, "runs", "traffic_highlight_output.mp4")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # --- 视频参数提取 ---
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 定义编解码器并创建 VideoWriter 对象
    # 'mp4v' 是最通用的格式；如果你在 Linux 下建议使用 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"开始处理，结果将保存至: {output_path}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 2. YOLO26 推理
        # retina_masks=True 保证保存的视频边缘平滑
        results = model.predict(frame, conf=0.3, retina_masks=True, verbose=False)
        
        # 初始化掩码图
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 3. 提取并合并所有目标的掩码
        if results[0].masks is not None:
            # 获取掩码数据并转换为 uint8 格式
            masks = results[0].masks.data.cpu().numpy()
            for mask in masks:
                # 将 mask 调整到与原图一致的大小 (如果没开 retina_masks)
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask, (width, height))
                combined_mask = cv2.bitwise_or(combined_mask, (mask * 255).astype(np.uint8))

        # --- 4. 图像特效处理 ---
        # 背景变暗处理：这里采用降低亮度（除以3），比全灰度更具动感
        background = frame // 3 
        
        # 提取彩色目标
        foreground = cv2.bitwise_and(frame, frame, mask=combined_mask)
        
        # 提取背景区域并合成
        inv_mask = cv2.bitwise_not(combined_mask)
        background_part = cv2.bitwise_and(background, background, mask=inv_mask)
        
        result_frame = cv2.add(foreground, background_part)

        # 5. 写入视频文件
        out.write(result_frame)

        # (可选) 预览处理进度
        # cv2.imshow("Processing...", cv2.resize(result_frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("视频处理完成并已安全保存。")

if __name__ == "__main__":
    process_and_save_video()