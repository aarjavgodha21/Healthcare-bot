## DiagnoGenie - DiagnoGenie Award-Winning Healthcare AI


An advanced healthcare AI system with:
- Modern home screen (Chatbot & Image Upload)
- Chatbot for medical Q&A (safe, rule-based, or LLM if configured)
- Image upload for X-ray/MRI with diagnosis, confidence, suggestions, and 4+ visual outputs (Grad-CAM, overlays, etc.)
- Beautiful, aesthetic UI with Tailwind and gradients


### Features
- **Home Screen**: Choose Chatbot or Image Upload
- **Chatbot**: Medical Q&A, suggestions, safety disclaimers
- **Image Upload**: Upload X-ray or MRI, get diagnosis, confidence, suggestions, and 4+ visual diagrams (Grad-CAM, overlays, saliency, etc.)
- **Tabular Diagnosis**: Predict from demographics, labs, symptoms
- **Modern UI**: Award-winning look, gradients, responsive, Tailwind
- **Robust Backend**: FastAPI, PyTorch, torchxrayvision, Grad-CAM


### Quick Start (Windows/PowerShell)

#### 1. Setup Python Environment
```bash
python -m venv .venv
.venv\Scripts\activate
cd backend
pip install -r requirements.txt
```

#### 2. Train the Model (if not already trained)
```bash
cd backend
python train_model.py
```

#### 3. Start Ollama (optional, for better chatbot)
```bash
ollama serve
```

#### 4. Start the API Server
```bash
cd backend
uvicorn api:app --reload --port 8001
```

API endpoints:
- **Health**: GET http://localhost:8001/health
- **Schema**: GET http://localhost:8001/schema
- **Predict**: POST http://localhost:8001/predict
- **Image Diagnosis**: POST http://localhost:8001/image-diagnosis (file upload)
- **Chatbot**: POST http://localhost:8001/chat


#### 5. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 in your browser.


### API Usage


#### Image Diagnosis (X-ray/MRI)
curl -X POST http://localhost:8001/image-diagnosis \
  -F "file=@Tests/covid.jpeg" \
  -F "image_type=auto"

#### Chatbot
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What does my X-ray show?"}]}'

### Model Details
- **Algorithm**: RandomForestClassifier (200 trees, max_depth=20)
- **Features**: 15 numeric, 2 categorical, 1 text (TF-IDF with 500 features)
- **Preprocessing**: Median imputation, standard scaling, one-hot encoding
- **Performance**: 100% accuracy on synthetic test data


### Project Structure
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
│   ├── src/App.jsx         # Legacy tabular diagnosis
│   ├── src/main.jsx        # Routing
│   ├── package.json        # Dependencies
│   └── tailwind.config.js  # Styling config
├── Tests/                  # Test images (covid.jpeg, etc.)
└── README.md               # This file
```


### Notes
- **Ollama**: For best chatbot experience, install Ollama and run `ollama pull llama3` or `ollama pull medllama`. Set `OLLAMA_MODEL=medllama` environment variable if using a medical model.
- All diagnosis and suggestions are for educational support only; not a substitute for professional medical judgment.
- Image upload returns at least 4 visual diagrams (original, Grad-CAM, overlays, saliency/edges).
- UI is fully responsive and modern.
- CORS is enabled for local development.
