# TRAIT Backend 🔍
Backend files for TRAIT Senior Project.
## 📋 Overview
TRAIT Backend is a Python-based server application that provides API endpoints and backend functionality for the TRAIT project. This is part of a senior project implementation.
## ⚡ Prerequisites
- Python 3.x
- pip (Python package manager)
## 🔧 Installation
1. Clone the repository:
```bash
git clone https://github.com/logan-taggart/TRAIT-Back.git
cd TRAIT-Back
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. **For macOS users**: If you encounter security issues with the package, run:
```bash
xattr -rd com.apple.quarantine TRAIT-Back
```
## 🏃‍♂️ Usage
### Running the Server
Start the backend server using:
```bash
python3 run.py
```
The server will start on **port 5174**.
### 🔗 API Endpoints
Image Endpoints:
- `GET /image/` - Image health check endpoint
- `POST /image/detect-all` - Detect all logos in an image endpoint
- `POST /image/detect-specific` - Detect specific logos in an image endpoint
- `POST /image/cancel` - Cancel image processing endpoint
  
Video Endpoints:
- `GET /video/` - Video health check endpoint
- `GET /video/fetch-progress` - Status of current video processing endpoint
- `GET /video/fetch-processed-video` - Retrieve proccesed video endpoint
- `POST /video/detect-all` - Detect all logos in a video endpoint
- `POST /video/detect-specific` - Detect specific logos in a video endpoint
- `POST /video/cancel` - Cancel video processing endpoint
## ⚙️ Configuration
- Server runs on `localhost:5174` by default
## 📁 Project Structure
```
TRAIT-Back/
├── run.py           
├── app.py         
├── models/
|     ├── logo_detection.pt
|     └── model_load.py
├── routes/
│     ├── image_routes.py
│     └── video_routes.py
├── utils/
│     ├── cancel_process.py
│     ├── video_progress.py
│     ├── embed.py
│     ├── logo_detection_utils.py
│     ├── process_image.py
│     └── process_video.py
├── .gitignore
├── .dockerignore
├── Dockerfile
├── requirements.txt   
└── README.md     
```
## 🛠️ Development & Troubleshooting
### Common Issues
**Port already in use:**
- Check if another application is using port 5174
- Kill the process or change the port in the configuration

**Permission errors on macOS:**
- Run the `xattr` command mentioned in the installation section

**Module not found errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check your Python version compatibility
  
## 🏗️ Technology Stack
- **Python 3.x** - Main programming language
- **Flask, Ultralytics, Numpy, Transformers, FAISS, CV2...** - Main Python libraries
- **YOLOv8** - Computer Vision Model
## Additional
- **Author**: Logan Taggart, Caleb Stewart, Lane Keck
- **Link to Backend Repository**: [TRAIT-Back](https://github.com/logan-taggart/TRAIT-Back)
- **Link to Frontend Repository**: [TRAIT-Front](https://github.com/logan-taggart/TRAIT-Front)
---
*Last updated: June 1st, 2025*
