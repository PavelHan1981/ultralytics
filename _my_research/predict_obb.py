import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. 加载 OBB 预训练模型 (YOLO26 官方提供多种规格)
# 'n' 代表轻量化，适合边缘端实时测试
model = YOLO('_my_research/pre_trained/yolov8l-obb.pt') 

def run_yolo26_obb_demo():
    # 获取当前项目目录的绝对路径
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # 建议使用包含斜向车辆或俯拍工业零件的视频/图片
    source = "D:/Datasets/traffic_video/vecteezy_aerial-view-of-british-roads-and-traffic-passing-through_18137910.mp4" # 替换为你的视频路径

    # --- OBB 推理 ---
    # OBB 模式同样支持 track，这在越线检测中非常实用
    results = model.track(source=source, stream=True, persist=True, conf=0.3, show=False, save=True,project=os.path.join(project_dir, "runs", "obb"))

    for r in results:
        # 在窗口中实时显示预览（自带旋转框渲染）
        im0 = r.plot() 
        cv2.imshow("YOLO26 OBB Tracking", im0)

        # --- 核心：提取旋转框数据 ---
        if r.obb is not None:
            # r.obb.xyxyxyxy 是 OBB 的四个角点坐标: [N, 4, 2]
            # 这对于折线判定非常完美，因为你可以精确知道物体的每一个角是否触线
            corners = r.obb.xyxyxyxy.cpu().numpy() 
            
            # r.obb.xywhr 是中心点、宽高和旋转弧度: [N, 5]
            # 其中 r 是弧度制 (Radian)
            specs = r.obb.xywhr.cpu().numpy()
            
            ids = r.obb.id.cpu().numpy() if r.obb.id is not None else []

            for i, box_corners in enumerate(corners):
                obj_id = ids[i] if len(ids) > 0 else "N/A"
                print(f"ID: {obj_id} | 四角坐标: {box_corners.tolist()}")
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo26_obb_demo()