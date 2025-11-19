import os

os.environ['WANDB_DISABLED'] = 'true'
from ultralytics import YOLO

# model =YOLO("snn_yolov8l.yaml").load('69M_best.pt')
model=YOLO("snn_yolov8n.yaml")

print(model)

# 显示模型参数量、FLOPs等
# model.info()

#train
# model.train(data="coco.yaml",device=[7],epochs=100)  # train the model
# model.train(data="DUO.yaml",device=[0,1],epochs=300,batch=32,imgsz=640)  # train the model
model.train(data="DUO.yaml",device=[2,3],epochs=300,batch=64,imgsz=640,copy_paste=0.5)  # train the model


#TEST
# model = YOLO('runs/detect/train1/weights/last.pt')  # load a pretrained model (recommended for training)

