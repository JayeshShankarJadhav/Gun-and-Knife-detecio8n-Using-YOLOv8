from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="model/data.yaml",
    epochs=150,
    batch=6,
    imgsz=640,
    optimizer="Adam",
    project="runs/detect",
    name="Db",
    pretrained=True,
    patience=50,
    iou=0.7,
    overlap_mask=True,
    mask_ratio=4,
    deterministic=True,
    workers=8
)


model.train(
    data="model/data.yaml",
    epochs=150,
    batch=6,
    imgsz=640,
    optimizer="Adam",
    project="runs/detect",
    name="Haar",
    pretrained=True,
    patience=50,
    iou=0.7,
    overlap_mask=True,
    mask_ratio=4,
    deterministic=True,
    workers=8
)


model.train(
    data="model/data.yaml",
    epochs=150,
    batch=6,
    imgsz=640,
    optimizer="Adam",
    project="runs/detect",
    name="Haar_Compressed",
    pretrained=True,
    patience=50,
    iou=0.7,
    overlap_mask=True,
    mask_ratio=4,
    deterministic=True,
    workers=8
)

model.train(
    data="model/data.yaml",
    epochs=150,
    batch=6,
    imgsz=640,
    optimizer="Adam",
    project="runs/detect",
    name="Symlet",
    pretrained=True,
    patience=50,
    iou=0.7,
    overlap_mask=True,
    mask_ratio=4,
    deterministic=True,
    workers=8
)



model.train(
    data="model/data.yaml",
    epochs=150,
    batch=6,
    imgsz=640,
    optimizer="Adam",
    project="runs/detect",
    name="Normal",
    pretrained=True,
    patience=50,
    iou=0.7,
    overlap_mask=True,
    mask_ratio=4,
    deterministic=True,
    workers=8
)


model.train(
    data="model/data.yaml",
    epochs=150,
    batch=6,
    imgsz=640,
    optimizer="Adam",
    project="runs/detect",
    name="Normal_Compressed",
    pretrained=True,
    patience=50,
    iou=0.7,
    overlap_mask=True,
    mask_ratio=4,
    deterministic=True,
    workers=8
)