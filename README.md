# EcoSwap-AI

A machine learning model that identifies any household object and recommends eco friednly alternatives. Built after a comparitive study between Vision Transformers and Convulutional Neural Network. 

📦 Dataset Size : 40,145 images, 63 household-object categories
🧠 Models Compared : ViT-Base-Patch16-224-In21k vs YOLOv11-cls
⚙ Training Epochs : 50 per configuration
🧪 Metrics Used	: Accuracy, Macro-F1, Macro-Recall, Confusion Matrix
🚀 Best Model	: Vision Transformer (ViT-Base-Patch16-224-In21k)

📊 Results

For ViT-Base Patch16:
Accuracy : 0.9053	
Macro-Recall : 0.8530
Macro-F1 : 0.8505

For YOLOv11: 
Accuracy : 0.8816	
Macro-Recall : 0.8401
Macro-F1 : 0.8430	

Key Observation:
Transformers outperform CNNs in household-object recognition due to stronger global attention modeling.
YOLO remains advantageous for speed, deployment, and edge-device application.
