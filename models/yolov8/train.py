from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = YOLO('yolov8s.pt')
model.train(
    data="data.yaml",
    batch=8, epochs=50
)
