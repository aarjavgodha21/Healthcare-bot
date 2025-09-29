# 🏥 DiagnoGenie - AI-Powered Medical Platform

A comprehensive healthcare AI platform that provides medical imaging analysis, symptom consultation, and lab report interpretation using advanced machine learning models.

## 📹 **Submissions**

### **Demonstration Video**
🎥 **Complete System Demo**: [Watch DiagnoGenie in Action](https://drive.google.com/file/d/1Hj_ZwEGGdzQHYF2DPedsKZG1B30ituy3/view?usp=drive_link)

*Comprehensive demonstration showcasing all medical AI features including chatbot consultation, X-ray analysis, MRI diagnosis, and lab report processing.*

### **Project Presentation**
📊 **Technical Presentation**: [DiagnoGenie Project Overview](https://drive.google.com/file/d/1_o05zHEaKLaisMADlgLAZQbR227rdDUK/view?usp=sharing)

*Detailed presentation covering architecture, technology stack, AI models, and implementation approach.*

## ✨ Features

### 🔬 Medical Imaging Analysis
- **X-ray Analysis**: Pneumonia, COVID-19, fractures, and 14+ pathology detection using CheXNet
- **MRI Brain Imaging**: Brain tumor, stroke, multiple sclerosis detection  
- **Advanced Visualizations**: AI attention maps, anatomical edge detection, pathology heatmaps

### 💬 AI Medical Consultation
- **Multilingual Support**: English and Hindi medical consultations
- **Symptom Analysis**: Comprehensive differential diagnosis and treatment recommendations
- **Voice Integration**: Text-to-speech for accessibility
- **Structured Reports**: Professional medical consultation format

### 📋 Lab Report Processing
- **PDF Analysis**: Blood reports, urine tests, infectious disease markers
- **Smart Extraction**: Automated value parsing and clinical interpretation
- **Risk Assessment**: Flagging abnormal values with clinical context

### 🎨 Modern Interface
- **WebGL Animations**: Interactive orb background in chatbot
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Medical UI/UX**: Professional healthcare-focused design system
- **Accessibility**: Screen reader compatible, keyboard navigation

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Ollama** (optional, for enhanced AI chat)

### Installation & Setup

📖 **For detailed setup instructions and troubleshooting, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

#### 1. Python Backend Setup
```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Train the model (if not already trained)
python train_model.py
```

#### 2. Frontend Setup
```powershell
cd frontend
npm install
```

#### 3. Start Services (Individual Scripts)

**🚀 Use the convenient PowerShell scripts for easy startup:**

```powershell
# Start backend API server
.\start-backend.ps1     # Runs on http://localhost:8001

# Start frontend development server  
.\start-frontend.ps1    # Runs on http://localhost:5173

# Start Ollama (optional for enhanced AI chat)
.\start-ollama.ps1      # Ollama service for LLM integration
```

**Alternative - Manual startup:**
```powershell
# Frontend
cd frontend ; npm run dev

# Backend (in separate terminal)
cd backend ; .venv\Scripts\activate ; uvicorn api:app --reload --port 8001

# Ollama (in separate terminal, optional)
ollama serve
```

#### 4. Access the Application
Open http://localhost:5173 in your browser to access DiagnoGenie.

## 🛠️ Technology Stack

### Data Ingestion & Processing
- **Python 3.8+**: Core backend language
- **FastAPI**: High-performance web API framework
- **OpenCV**: Computer vision and image processing
- **PyPDF2**: PDF lab report extraction
- **Pillow**: Image manipulation and format conversion

### AI Models & Machine Learning
- **PyTorch**: Deep learning framework for medical imaging
- **torchxrayvision**: CheXNet model for X-ray analysis (14+ pathologies)
- **scikit-learn**: RandomForest classifier for symptom diagnosis
- **TensorFlow**: MRI analysis and custom medical models
- **Grad-CAM**: AI explainability and attention visualization

### Backend API & Services
- **FastAPI + Uvicorn**: RESTful API with async capabilities
- **Pydantic**: Data validation and serialization
- **CORS**: Cross-origin resource sharing for web integration
- **File Upload**: Multi-format medical image support
- **Ollama Integration**: Optional LLM for enhanced medical consultations

### Frontend UI & Experience
- **React 18**: Modern component-based UI framework
- **Vite**: Fast development and optimized builds
- **Tailwind CSS**: Utility-first styling with medical design system
- **OGL (WebGL)**: 3D medical visualizations and animations
- **Responsive Design**: Mobile-first medical interface

## 📖 API Documentation

### Medical Imaging
```bash
POST /image-diagnosis
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPEG, PNG, DICOM, NIfTI)
- image_type: 'xray', 'mri', or 'auto'

Response: Diagnosis, confidence, visualizations, suggestions
```

### AI Chat Consultation  
```bash
POST /chat
Content-Type: application/json

Body:
{
  "messages": [{"role": "user", "content": "symptom description"}],
  "language": "en" | "hi"
}

Response: Structured medical consultation
```

### Lab Report Analysis
```bash
POST /lab-report
Content-Type: multipart/form-data

Parameters:
- file: PDF blood/urine report

Response: Parsed values, interpretations, recommendations
```

### Example Usage
```bash
# X-ray analysis
curl -X POST http://localhost:8001/image-diagnosis \
  -F "file=@Tests/xray1.jpeg" \
  -F "image_type=xray"

# Medical consultation
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I have fever and cough"}], "language": "en"}'
```

## 🛠️ Architecture

```
Healthcare/
├── backend/                # Backend (FastAPI, models, configs)
│   ├── api.py              # FastAPI backend
│   ├── train_model.py      # Training pipeline
│   ├── diagnogenie_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_config.json
│   ├── diagno_genie_demo_10000.csv
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/Home.jsx        # Home screen
│   ├── src/Chatbot.jsx     # Chatbot page
│   ├── src/ImageUpload.jsx # Image upload page
│   ├── src/App.jsx         # Main application
│   ├── src/main.jsx        # Entry point
│   ├── package.json        # Dependencies
│   └── tailwind.config.js  # Styling config
├── Tests/                  # Test images (covid.jpeg, etc.)
└── README.md               # This file
```

## 🎯 Model Performance

### X-ray Analysis (CheXNet)
- **Pathologies Detected**: 14+ conditions including pneumonia, COVID-19, fractures
- **Visualization**: 4+ medical visualizations per analysis
- **Confidence Scoring**: Probability-based risk assessment

### Symptom Analysis  
- **Algorithm**: RandomForestClassifier (200 trees, max_depth=20)
- **Features**: 15 numeric, 2 categorical, 1 text (TF-IDF with 500 features)
- **Preprocessing**: Median imputation, standard scaling, one-hot encoding

## 🔒 Security & Compliance

- **Privacy**: No patient data stored permanently
- **Security**: Input validation, file type restrictions, CORS protection
- **Medical Disclaimer**: Educational use only, not for clinical diagnosis
- **HIPAA Awareness**: Designed with healthcare privacy principles

## 🌐 Multilingual Support

### Languages Supported
- **English**: Complete medical terminology and consultations
- **Hindi (हिंदी)**: Full medical consultation in Devanagari script
- **Auto-Detection**: Automatically detects user's preferred language

## 📱 Supported Formats

### Medical Images
- **X-rays**: JPEG, PNG, DICOM, BMP, TIFF
- **MRI**: NIfTI (.nii, .nii.gz), JPEG, PNG
- **Maximum Size**: 8MB per file

### Lab Reports
- **Format**: PDF (text-based, not scanned images)
- **Types**: Blood work, urine tests, infectious disease markers
- **Extraction**: Automated parsing of common lab values

## 🧪 Development & Extension

### Adding New Medical Conditions
1. Update training data in `backend/train_model.py`
2. Add condition-specific responses in `backend/api.py`
3. Update UI components in `frontend/src/components/`

### Customizing Medical UI
- Modify `frontend/tailwind.config.js` for medical color schemes
- Update design system in `frontend/src/components/DesignSystem.jsx`
- Add new medical icons and visualizations

### Performance Optimization
- Models are lazy-loaded for faster startup
- Image processing optimized for web delivery
- Caching implemented for ML model predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/medical-enhancement`
3. Follow medical coding best practices and ethical AI guidelines
4. Add comprehensive tests for new medical features
5. Submit a pull request with detailed medical context

## ⚖️ Medical Disclaimer

**⚠️ IMPORTANT: This software is for educational and research purposes only.**

- **Not for Clinical Use**: Not intended for clinical diagnosis or treatment decisions
- **Professional Consultation**: Always consult qualified healthcare professionals for medical concerns
- **AI Limitations**: AI predictions are supportive tools, not replacements for medical expertise
- **Emergency Care**: Seek immediate medical attention for medical emergencies
- **No Liability**: This software provides no medical guarantees or liability coverage

## 📞 Support & Community

### Getting Help
- **Documentation**: Comprehensive guides in this README and [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Issues**: Create GitHub issues for bugs or feature requests
- **Medical Questions**: Include anonymized medical context for better support

### Contributing Guidelines
- Follow ethical AI practices in healthcare
- Maintain patient privacy and data security
- Test thoroughly with diverse medical scenarios
- Document medical assumptions and limitations

---

**🏥 DiagnoGenie** - Empowering healthcare education with responsible AI  
*Built with ❤️ for medical research and education*

**Version 1.0.0** | **License**: Educational Use Only | **Last Updated**: 2025