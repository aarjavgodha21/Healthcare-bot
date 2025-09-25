# 🏥 DiagnoGenie - AI-Powered Medical Platform## DiagnoGenie - DiagnoGenie Award-Winning Healthcare AI



A comprehensive healthcare AI platform that provides medical imaging analysis, symptom consultation, and lab report interpretation using advanced machine learning models.

An advanced healthcare AI system with:

## ✨ Features- Modern home screen (Chatbot & Image Upload)

- Chatbot for medical Q&A (safe, rule-based, or LLM if configured)

### 🔬 Medical Imaging Analysis- Image upload for X-ray/MRI with diagnosis, confidence, suggestions, and 4+ visual outputs (Grad-CAM, overlays, etc.)

- **X-ray Analysis**: Pneumonia, COVID-19, fractures, and 14+ pathology detection using CheXNet- Beautiful, aesthetic UI with Tailwind and gradients

- **MRI Brain Imaging**: Brain tumor, stroke, multiple sclerosis detection

- **Advanced Visualizations**: AI attention maps, anatomical edge detection, pathology heatmaps

### Features

### 💬 AI Medical Consultation- **Home Screen**: Choose Chatbot or Image Upload

- **Multilingual Support**: English and Hindi medical consultations- **Chatbot**: Medical Q&A, suggestions, safety disclaimers

- **Symptom Analysis**: Comprehensive differential diagnosis and treatment recommendations- **Image Upload**: Upload X-ray or MRI, get diagnosis, confidence, suggestions, and 4+ visual diagrams (Grad-CAM, overlays, saliency, etc.)

- **Voice Integration**: Text-to-speech for accessibility- **Tabular Diagnosis**: Predict from demographics, labs, symptoms

- **Structured Reports**: Professional medical consultation format- **Modern UI**: Award-winning look, gradients, responsive, Tailwind

- **Robust Backend**: FastAPI, PyTorch, torchxrayvision, Grad-CAM

### 📋 Lab Report Processing

- **PDF Analysis**: Blood reports, urine tests, infectious disease markers

- **Smart Extraction**: Automated value parsing and clinical interpretation### Quick Start (Windows/PowerShell)

- **Risk Assessment**: Flagging abnormal values with clinical context

#### 1. Setup Python Environment

### 🎨 Modern Interface```bash

- **WebGL Animations**: Interactive orb background in chatbotpython -m venv .venv

- **Responsive Design**: Works on desktop, tablet, and mobile.venv\Scripts\activate

- **Medical UI/UX**: Professional healthcare-focused design systemcd backend

- **Accessibility**: Screen reader compatible, keyboard navigationpip install -r requirements.txt

```

## 🚀 Quick Start

#### 2. Train the Model (if not already trained)

### Prerequisites```bash

- **Python 3.8+** with pipcd backend

- **Node.js 16+** with npmpython train_model.py

- **Ollama** (optional, for enhanced AI chat)```



### Installation & Setup#### 3. Start Ollama (optional, for better chatbot)

```bash

#### 1. Python Backend Setupollama serve

```powershell```

# Create virtual environment

python -m venv .venv#### 4. Start the API Server

.venv\Scripts\activate```bash

cd backend

# Install backend dependenciesuvicorn api:app --reload --port 8001

cd backend```

pip install -r requirements.txt

API endpoints:

# Train the model (if not already trained)- **Health**: GET http://localhost:8001/health

python train_model.py- **Schema**: GET http://localhost:8001/schema

```- **Predict**: POST http://localhost:8001/predict

- **Image Diagnosis**: POST http://localhost:8001/image-diagnosis (file upload)

#### 2. Frontend Setup- **Chatbot**: POST http://localhost:8001/chat

```powershell

cd frontend

npm install#### 5. Start the Frontend

``````bash

cd frontend

#### 3. Start Servicesnpm install

```powershellnpm run dev

# Option 1: Start all services at once```

npm run startOpen http://localhost:5173 in your browser.



# Option 2: Start services individually

npm run start-frontend  # Runs on http://localhost:5173### API Usage

npm run start-backend   # Runs on http://localhost:8000

npm run start-ollama    # Optional: Enhanced AI chat

```#### Image Diagnosis (X-ray/MRI)

curl -X POST http://localhost:8001/image-diagnosis \

#### 4. Access the Application  -F "file=@Tests/covid.jpeg" \

Open http://localhost:5173 in your browser to access DiagnoGenie.  -F "image_type=auto"



## 🛠️ Architecture#### Chatbot

curl -X POST http://localhost:8001/chat \

```  -H "Content-Type: application/json" \

Healthcare/  -d '{"messages": [{"role": "user", "content": "What does my X-ray show?"}]}'

├── frontend/           # React + Vite + Tailwind CSS

│   ├── src/### Model Details

│   │   ├── App.jsx            # Main application- **Algorithm**: RandomForestClassifier (200 trees, max_depth=20)

│   │   ├── Chatbot.jsx        # AI medical consultation- **Features**: 15 numeric, 2 categorical, 1 text (TF-IDF with 500 features)

│   │   ├── ImageUpload.jsx    # Medical imaging analysis- **Preprocessing**: Median imputation, standard scaling, one-hot encoding

│   │   └── components/        # Reusable UI components- **Performance**: 100% accuracy on synthetic test data

│   └── public/

├── backend/            # Python FastAPI

│   ├── api.py                 # Main API server### Project Structure

│   ├── requirements.txt       # Python dependencies```

│   └── *.pkl                  # Pre-trained ML modelsHealthcare/

└── Tests/              # Sample medical files├── backend/                # Backend (FastAPI, models, configs)

```│   ├── api.py              # FastAPI backend

│   ├── train_model.py      # Training pipeline

### 🔧 Technology Stack│   ├── diagnogenie_model.pkl

│   ├── preprocessor.pkl

**Frontend:**│   ├── feature_config.json

- React 18 + Vite (Fast development and building)│   ├── diagno_genie_demo_10000.csv

- Tailwind CSS (Utility-first styling)│   └── requirements.txt    # Python dependencies

- OGL (WebGL animations)├── frontend/               # React frontend

- Medical design system components│   ├── src/Home.jsx        # Home screen

│   ├── src/Chatbot.jsx     # Chatbot page

**Backend:**│   ├── src/ImageUpload.jsx # Image upload page

- FastAPI (High-performance Python API)│   ├── src/App.jsx         # Legacy tabular diagnosis

- PyTorch + torchxrayvision (Medical imaging AI)│   ├── src/main.jsx        # Routing

- scikit-learn (Symptom prediction models)│   ├── package.json        # Dependencies

- PIL + OpenCV (Image processing)│   └── tailwind.config.js  # Styling config

├── Tests/                  # Test images (covid.jpeg, etc.)

**AI Models:**└── README.md               # This file

- CheXNet DenseNet-121 (X-ray pathology detection)```

- Custom MRI brain analysis (Rule-based + ML)

- Tabular ML models (Symptom → diagnosis prediction)

### Notes

## 📖 API Documentation- **Ollama**: For best chatbot experience, install Ollama and run `ollama pull llama3` or `ollama pull medllama`. Set `OLLAMA_MODEL=medllama` environment variable if using a medical model.

- All diagnosis and suggestions are for educational support only; not a substitute for professional medical judgment.

### Medical Imaging- Image upload returns at least 4 visual diagrams (original, Grad-CAM, overlays, saliency/edges).

```bash- UI is fully responsive and modern.

POST /image-diagnosis- CORS is enabled for local development.

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
curl -X POST http://localhost:8000/image-diagnosis \
  -F "file=@Tests/xray1.jpeg" \
  -F "image_type=xray"

# Medical consultation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I have fever and cough"}], "language": "en"}'
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
- **Documentation**: Comprehensive guides in this README
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