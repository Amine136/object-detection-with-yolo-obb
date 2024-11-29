# Object Detection with YOLO and Oriented Bounding Boxes (OBB)

This project is designed to detect white objects in images, similar to those provided in the `testset` folder. It leverages **YOLOv8** with oriented bounding boxes to handle object detection at various angles, providing more precise and adaptable results compared to traditional rectangular detection.

The project includes a **user-friendly dashboard** built with **Streamlit**, which allows users to:
- Upload images for analysis.
- Display detection results with annotated images.
- View the total number of detections.

---

## Installation and Usage

### Prerequisites
Make sure you have Python installed on your system. 

### Steps to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/Amine136/object-detection-with-yolo-obb.git
   cd object-detection-with-yolo-obb

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


3. Run the application:
   ```bash
   streamlit run main.py
