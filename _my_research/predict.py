import os
from ultralytics import YOLO

# 1. 加载模型 (自动下载预训练权重)
# 'n' 代表 Nano 版本；可选 's', 'm', 'l', 'x'
model = YOLO('_my_research/pre_trained/yolo26n.pt') 

def run_yolo26_demos():
    # 获取当前项目目录的绝对路径
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # video_path = "D:/Datasets/traffic_video/vecteezy_aerial-view-of-british-roads-and-traffic-passing-through_18137910.mp4" # 替换为你的视频路径
    video_path = "D:/Datasets/traffic_video/vecteezy_slow-motion-heavy-traffic-in-the-center-of-bangkok-around_24210406.mov" # 替换为你的视频路径

    # --- 模式 A: 标准物体检测 (Object Detection) ---
    # 验证模型是否能正确识别出人(person)或车(car)
    print("正在执行：标准检测模式...")
    # 标准检测模式，save=True 保存结果，使用绝对路径指定保存到当前项目目录的 runs/detect/
    results = model.predict(source=video_path, conf=0.25, save=True, show=False, project=os.path.join(project_dir, "runs", "detect"))
    
    # --- 模式 B: 实时多目标跟踪 (Tracking) ---
    # 【越线检测的核心】必须开启 persist=True 以锁定目标 ID
    print("正在执行：多目标跟踪模式...")
    # tracker可选: 'bytetrack.yaml' (推荐) 或 'botsort.yaml'
    # 【注意】如果是静态场景（如监控视频），使用 ByteTrack 即可。
    # 如果相机处于运动状态（如无人机或车载），推荐使用 BoT-SORT，它包含相机运动补偿。
    # 添加 save=True 确保结果被保存，使用绝对路径指定保存到当前项目目录的 runs/track/
    track_results = model.track(source=video_path, conf=0.3, persist=True, tracker="bytetrack.yaml", show=False, save=True, project=os.path.join(project_dir, "runs", "track"))

    # --- 模式 C: 异步流式推理 (Generator Mode) ---
    # 适合处理长视频，避免内存溢出，方便接入你的越线逻辑
    # print("正在执行：流式处理模式...")
    # gen_results = model.track(source=video_path, stream=True, persist=True)
    # for r in gen_results:
    #     # 在这里插入你的越线判定逻辑 (Cross Product Logic)
    #     boxes = r.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
    #     ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else []
    #     print(f"当前帧检测到 {len(boxes)} 个目标，IDs: {ids}")
    #     # break # 测试用途，仅跑一帧

    # --- 模式 D: 模型导出与量化 (Export) ---
    # 用于生产环境部署，YOLO26 导出 TensorRT 或 ONNX 时无需处理 NMS 算子
    # print("正在执行：模型导出...")
    # model.export(format="onnx", imgsz=640, dynamic=False)
    # model.export(format="engine", device=0) # 导出为 TensorRT

if __name__ == "__main__":
    run_yolo26_demos()