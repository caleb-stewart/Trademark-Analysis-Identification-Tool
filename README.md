# TRAIT - Logo Detection System 🎯

Complete full-stack application for AI-powered logo detection in images and videos.

## 🚀 Quick Start

### Installation & Running
1. **Clone the repository:**
```bash
git clone https://github.com/caleb-stewart/Trademark-Analysis-Identification-Tool.git
cd Trademark-Analysis-Identification-Tool
cd Project
```

2. **Start Backend (Port 5174):**
```bash
cd TRAIT-Back
pip install -r requirements.txt
python3 run.py
```

3. **Start Frontend (Port 5173):**
```bash
cd ../TRAIT-Front
npm install
npm run start
```

4. **Access the application at:** `http://localhost:5173` or on the desktop using Electron

## 📋 What is TRAIT?
TRAIT is an AI-powered logo detection system that can identify and locate logos in both images and videos using YOLOv8 computer vision technology.

### ✨ Features
- **Image Logo Detection** - Upload images and detect all or specific logos
- **Video Logo Detection** - Process videos with real-time progress tracking
- **Interactive Desktop/Web Interface** - User-friendly React frontend
- **RESTful API** - Comprehensive backend API for all operations

## 🏗️ Architecture
- **Frontend:** React application (Port 5173)
- **Backend:** Python Flask API (Port 5174)
- **AI Model:** YOLOv8 for logo detection
- **Libraries:** Ultralytics, OpenCV, NumPy, Transformers, FAISS, etc.

## 📁 Repository Structure
```
Trademark-Analysis-Identification-Tool/
├── Documents/
|      ├── Other/...
|      ├── Presentations/...
|      └── Sprint Reports/...
├── Misc/
|      ├── .ipynb_checkpoints/...
|      ├── YOLO Model/...
|      ├── test_scripts/...
|      └── index.html
├── Project/
|      ├── TRAIT-Back/...
|      ├── TRAIT-Front/...
├── .gitignore
└── README.md
```

## 🔗 Individual Repositories
- **Frontend Only:** [TRAIT-Front](https://github.com/logan-taggart/TRAIT-Front)
- **Backend Only:** [TRAIT-Back](https://github.com/logan-taggart/TRAIT-Back)

## 👥 Team
**Authors:** Logan Taggart, Caleb Stewart, Lane Keck  
**Project:** Senior Capstone Project

---
*TRAIT - Trademark Analysis and Identification Tool*  
*Updated June 7th, 2025*
