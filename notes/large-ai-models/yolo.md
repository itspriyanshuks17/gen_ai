# YOLO (You Only Look Once) - Developer Notes

## Introduction

YOLO is a real-time object detection system that predicts bounding boxes and class probabilities simultaneously in a single neural network pass.

### How YOLO Works
- **Single Shot Detection**: Process entire image once
- **Grid Division**: Divide image into S×S grid cells
- **Per Cell Prediction**: Each cell predicts B bounding boxes and C class probabilities
- **Confidence Score**: Object presence probability × IOU accuracy

### Architecture
```mermaid
flowchart LR
    A[Input Image 448x448] --> B[Darknet-53 Backbone]
    B --> C[Detection Head]
    C --> D[13x13x(5*B+C) Tensor]
    D --> E[Bounding Boxes]
    D --> F[Class Probabilities]
    D --> G[Confidence Scores]
```

### YOLO Versions

#### YOLOv1
- **Grid Size**: 7×7
- **Bounding Boxes**: 2 per cell
- **Classes**: 20 (Pascal VOC)
- **Limitations**: Limited accuracy, few objects per image

#### YOLOv2 (YOLO9000)
- **Improvements**: Batch normalization, anchor boxes, multi-scale training
- **Anchor Boxes**: Predefined box shapes for better localization
- **Dataset**: Combined COCO + ImageNet (9000 classes)

#### YOLOv3
- **Backbone**: Darknet-53 (53 convolutional layers)
- **Feature Pyramid**: Multi-scale detection
- **Classes**: 80 (COCO dataset)
- **Speed**: 45 FPS on Titan X

#### YOLOv4
- **Optimizations**: CSPDarknet53, PANet, data augmentation
- **Techniques**: Mosaic, Self-Adversarial Training
- **Performance**: State-of-the-art accuracy and speed

#### YOLOv5
- **Framework**: PyTorch implementation
- **Features**: Auto-anchoring, hyperparameter evolution
- **Ease of Use**: Simple training and deployment

### Code Example: YOLO Object Detection

```python
import cv2
import numpy as np

def load_yolo_model():
    """Load YOLO model and configuration"""
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # Load class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

def detect_objects(img, net, output_layers, classes, conf_threshold=0.5, nms_threshold=0.4):
    """Detect objects in image using YOLO"""
    height, width, channels = img.shape
    
    # Preprocess image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Initialize detection lists
    class_ids = []
    confidences = []
    boxes = []
    
    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Draw bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img

# Usage
net, classes, output_layers = load_yolo_model()
image = cv2.imread("image.jpg")
result = detect_objects(image, net, output_layers, classes)
cv2.imshow("YOLO Detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Training YOLO
- **Dataset Preparation**: Images with bounding box annotations
- **Data Augmentation**: Mosaic, random scaling, flipping
- **Loss Function**: Combination of localization, confidence, and classification losses
- **Optimization**: SGD with momentum, learning rate scheduling

### Evaluation Metrics
- **mAP (mean Average Precision)**: Primary metric for object detection
- **IOU (Intersection over Union)**: Overlap between predicted and ground truth boxes
- **Precision/Recall**: Trade-off analysis

### Advantages
- **Real-time Performance**: Fast inference (45+ FPS)
- **End-to-End Training**: Single network for detection
- **Global Context**: Sees entire image at once
- **Generalization**: Works well on various object types

### Limitations
- **Small Objects**: Struggles with very small objects
- **Group Detection**: Difficulty detecting closely grouped objects
- **Aspect Ratios**: Limited by anchor box design
- **Background Errors**: Can have false positives

### Applications
- **Autonomous Vehicles**: Pedestrian and vehicle detection
- **Security Systems**: Intruder detection, surveillance
- **Retail Analytics**: Customer behavior analysis
- **Medical Imaging**: Anomaly detection
- **Industrial Inspection**: Quality control, defect detection

### Hinglish Explanation
YOLO (You Only Look Once) real-time object detection system hai jo single neural network pass mein bounding boxes aur class probabilities predict karta hai.

**Kaise kaam karta hai**: Image ko S×S grid cells mein divide karta hai, har cell B bounding boxes aur C class probabilities predict karta hai.

**Versions**: YOLOv1 se lekar YOLOv5 tak, har version mein improvements (anchor boxes, multi-scale, etc.)

**Advantages**: Real-time performance, end-to-end training, global context.

**Limitations**: Small objects mein difficulty, group detection problems.

**Applications**: Autonomous vehicles, security systems, retail analytics.