<h1> Surface Defect Detection </h1>

<h3> Problem Statement</h3>
Machining of metal surfaces in the factories leads to different types of defects which somehow affects the quality and strength of the metals. 
It is important to identify such defective pieces and remove or modify them.

This project tries to solve the problem by identifying such defects using a Deep Learning Open CV model.

Designed and deployed a production-ready computer vision system to classify industrial steel surface defects using deep learning and API-based inference.

<h3>Overview</h3>

Built a multi-class image classification pipeline to detect 6 defect types:
Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, and Scratches.

<h3>Technical Implementation</h3>

Transfer Learning using MobileNetV2 (ImageNet pretrained)

Fine-tuned upper layers for texture-sensitive defect recognition

Implemented correct ImageNet preprocessing and feature alignment

Used GlobalAveragePooling2D + custom dense classification head

Achieved ~93% balanced validation accuracy with strong per-class recall

<h3>Evaluation & Optimization</h3>

Conducted confusion matrix and per-class F1-score analysis

Identified and corrected preprocessing inconsistencies

Improved minority class detection through controlled fine-tuning

Built stable inference pipeline with confidence thresholding

<h3>Engineering Stack </h3>

TensorFlow / Keras

OpenCV (headless for backend processing)

FastAPI (REST API for image upload prediction)

Docker (containerized deployment)

Real-time webcam inference module

<h3>Deployment Architecture</h3>

Client (Image Upload) → FastAPI → Preprocessing → Model Inference → JSON Response

The system is fully containerized and ready for cloud deployment.
