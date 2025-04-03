from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")

train_results = model.train(
    lr0 = 1,
    momentum = 0.7,
    data="coco8-seg.yaml",
    batch=16,
    epochs=100,  
    imgsz=640, 
    device="cuda",
    workers = 0,  
)
print(train_results)
# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("mi_imagen.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")

if __name__ == '__main__':
    main()