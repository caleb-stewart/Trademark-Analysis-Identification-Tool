# TRAIT Frontend 🔍
Frontend files for TRAIT Senior Project.

## 📋 Overview
TRAIT Frontend is a modern web application that provides the user interface for the TRAIT project. This is part of a senior project implementation.

## ⚡ Prerequisites
- Node.js
- npm (Node package manager)

## 🔧 Installation
1. Clone the repository:
```bash
git clone https://github.com/logan-taggart/TRAIT-Front.git
cd TRAIT-Front
```

2. Install dependencies:
```bash
npm install
```

3. **For macOS users**: If you encounter security issues with the package, run:
```bash
xattr -rd com.apple.quarantine TRAIT-Front
```

## 🏃‍♂️ Usage
### Running the Development Server
Start the frontend server using:
```bash
npm run frontend
```
Start the frontend server and desktop Electron application using:
```bash
npm run start
```

The application will start on **port 5173** and automatically open in your browser at `http://localhost:5173`.

### 🔗 Features
- **Image Upload & Processing** - Upload images for logo detection
- **Video Upload & Processing** - Upload videos for logo detection
- **Real-time Progress Tracking** - Monitor processing status
- **Interactive Results Display** - View detected logos with visual indicators

## ⚙️ Configuration
- Development server runs on `localhost:5173` by default

## 📁 Project Structure
```
TRAIT-Front/
├── public/
│     ├── TRAIT_logo_circle.png
│     └── vite.svg
├── src/
│     ├── assets/
│     |     └── react.svg
│     ├── components/
│     |        ├── videoComponents/
|     |        │         ├── ProgressBar.jsx
|     |        │         ├── VideoResultDetails.jsx
|     |        │         ├── VideoResultDisplay.jsx
|     |        │         └── VideoUploadSection.jsx
│     |        ├── ColorPalette.jsx
│     |        ├── ConfidenceThreshold.jsx
│     |        ├── DetectionOptions.jsx
│     |        ├── Header.jsx
│     |        ├── ImageUploadSection.jsx
│     |        ├── ProcessingButton.jsx
│     |        ├── ResultDetails.jsx
│     |        ├── ResultDisplay.jsx
│     |        ├── SearchSpecificLogoOptions.jsx
│     |        ├── ThresholdInput.jsx
│     |        ├── ToolTipDropDown.jsx
│     |        └── UploadSection.jsx
│     ├── App.css
│     ├── App.jsx
│     ├── index.css
│     └── main.jsx
├── index.html
├── main.js
├── package.json     
├── package-lock.json
├── vite.config.js
├── .gitignore
├── .dockerignore
├── Dockerfile
└── README.md  
```

## 🛠️ Development & Troubleshooting

### Common Issues
**Port already in use:**
- Check if another application is using port 5173
- Kill the process or change the port in the configuration

**Permission errors on macOS:**
- Run the `xattr` command mentioned in the installation section

**Module not found errors:**
- Ensure all dependencies are installed: `npm install`
- Clear npm cache: `npm cache clean --force`

**Build failures:**
- Check Node.js version compatibility
- Delete `node_modules` and run `npm install` again

## 🏗️ Technology Stack
- **React** - Frontend framework
- **Vite** - Build Tool
- **Electron** - Desktop Application Framework Built on Chromium
- **HTML5/CSS3** - Markup and styling
- **JavaScript** - Programming language

## 🔗 Backend Integration
This frontend application works in conjunction with the TRAIT Backend:
- **Backend Repository**: [TRAIT-Back](https://github.com/logan-taggart/TRAIT-Back)
- **API Communication**: RESTful API calls to backend endpoints

## Additional
- **Author**: Logan Taggart, Caleb Stewart, Lane Keck
- **Frontend Repository**: [TRAIT-Front](https://github.com/logan-taggart/TRAIT-Front)
- **Backend Repository**: [TRAIT-Back](https://github.com/logan-taggart/TRAIT-Back)

---
*Last updated: June 1st, 2025*
