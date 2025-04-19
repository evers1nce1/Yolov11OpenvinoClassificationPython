from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="openvino", int8=True, imgsz=416, device='cpu', dynamic=True, data="../datasets/data2.yaml")
