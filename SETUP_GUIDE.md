# 🛠️ DiagnoGenie Setup Guide

## 📋 **Prerequisites**

### **System Requirements**
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+
- **RAM**: Minimum 8GB (16GB recommended for optimal AI performance)
- **Storage**: At least 5GB free space for models and dependencies
- **Internet**: Required for initial setup and model downloads

### **Required Software**
1. **Python 3.8 or higher** (3.10+ recommended)
   - Download from [python.org](https://python.org)
   - Ensure `pip` is included in installation
   - Verify: `python --version` and `pip --version`

2. **Node.js 16.0 or higher**
   - Download from [nodejs.org](https://nodejs.org)
   - Includes npm package manager
   - Verify: `node --version` and `npm --version`

3. **Git** (for cloning repository)
   - Download from [git-scm.com](https://git-scm.com)
   - Verify: `git --version`

## 🚀 **Installation Steps**

### **1. Clone the Repository**
```bash
git clone https://github.com/aarjavgodha21/Healthcare-bot.git
cd Healthcare-bot
```

### **2. Backend Setup**
```bash
# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install Python dependencies
cd backend
pip install -r requirements.txt

# Train the ML model (first-time setup)
python train_model.py

# Return to project root
cd ..
```

### **3. Frontend Setup**
```bash
# Install Node.js dependencies
cd frontend
npm install

# Return to project root
cd ..
```

### **4. Optional: Enhanced AI Chat Setup**
For improved medical consultations, install Ollama:

#### **Windows:**
```powershell
# Download and install Ollama from: https://ollama.com/download/windows
# After installation, pull medical model:
ollama pull llama3.2:1b
```

#### **macOS:**
```bash
# Install via Homebrew
brew install ollama
ollama pull llama3.2:1b
```

#### **Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
```

## ▶️ **Running DiagnoGenie**

### **Method 1: All Services (Recommended)**
```bash
# Start all services with one command
npm run start
```
This will display setup instructions and URLs for each service.

### **Method 2: Individual Services**
Open **3 separate terminals** and run:

#### **Terminal 1: Backend Server**
```bash
cd Healthcare-bot
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
.\start-backend.ps1  # Windows
# ./start-backend.sh  # macOS/Linux
```

#### **Terminal 2: Frontend Application**
```bash
cd Healthcare-bot
.\start-frontend.ps1  # Windows
# ./start-frontend.sh  # macOS/Linux
```

#### **Terminal 3: Enhanced AI Chat (Optional)**
```bash
.\start-ollama.ps1  # Windows
# ./start-ollama.sh  # macOS/Linux
```

### **Access the Application**
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs

## 🔧 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. Python Virtual Environment Issues**
```bash
# If activation fails, try:
python -m venv --clear .venv
# Then repeat activation steps
```

#### **2. Model Training Fails**
```bash
# Ensure you're in the backend directory with activated environment
cd backend
python -c "import torch; print('PyTorch version:', torch.__version__)"
# If PyTorch not found, reinstall:
pip install torch torchvision torchxrayvision
```

#### **3. Frontend Build Issues**
```bash
# Clear npm cache and reinstall
cd frontend
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

#### **4. Port Already in Use**
- **Backend (8001)**: Change port in `start-backend.ps1` to `--port 8002`
- **Frontend (5173)**: Vite will automatically use next available port

#### **5. CORS Issues**
- Ensure backend is running before frontend
- Check that frontend is accessing `http://127.0.0.1:8001` (not localhost)

### **Performance Optimization**

#### **For Better AI Performance:**
1. **GPU Support** (Optional):
   ```bash
   # Install CUDA PyTorch for GPU acceleration
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Memory Management**:
   - Close other applications during medical image analysis
   - Consider increasing virtual memory if system has limited RAM

#### **For Development:**
```bash
# Enable debug mode for detailed error logs
export DEBUG=1  # Linux/macOS
set DEBUG=1     # Windows
```

## 🧪 **Testing the Installation**

### **1. Test Backend Health**
```bash
curl http://localhost:8001/health
# Expected: {"status": "ok"}
```

### **2. Test Medical AI Models**
1. Open http://localhost:5173
2. Try the **Chatbot** with: "I have fever and cough"
3. Upload a test image in **Image Analysis**
4. Upload a PDF in **Lab Report Analysis**

### **3. Verify All Features**
- ✅ Medical chatbot responds with structured consultation
- ✅ Image upload shows AI analysis and visualizations  
- ✅ Lab report extraction displays parsed values
- ✅ WebGL orb animation appears in chatbot
- ✅ Language switching works (English ↔ Hindi)

## 📦 **Project Structure Verification**
After setup, your directory should look like:
```
Healthcare-bot/
├── .venv/                  # Python virtual environment
├── frontend/
│   ├── node_modules/       # Node.js dependencies
│   └── dist/              # Built frontend (after npm run build)
├── backend/
│   ├── diagnogenie_model.pkl    # Trained ML model
│   ├── preprocessor.pkl         # Data preprocessors
│   └── feature_config.json      # Model configuration
└── Tests/                  # Sample medical files for testing
```

## 🆘 **Getting Help**

### **If Setup Fails:**
1. **Check Prerequisites** - Ensure Python 3.8+, Node.js 16+ are installed
2. **Review Error Messages** - Look for missing dependencies or permission issues
3. **Clean Installation** - Delete `.venv` and `node_modules`, start fresh
4. **Platform-Specific Issues** - Check OS-specific commands above

### **For Medical AI Issues:**
1. **Model Files Missing** - Run `python train_model.py` in backend directory
2. **Image Analysis Fails** - Verify PyTorch and torchxrayvision installation
3. **Chat Not Responding** - Check if Ollama is running (optional feature)

### **System Resources:**
- **Minimum**: 4GB RAM, 2GB free storage
- **Recommended**: 8GB+ RAM, 5GB+ free storage
- **Optimal**: 16GB RAM, SSD storage, dedicated GPU

---

🎉 **Congratulations!** Your DiagnoGenie Healthcare AI platform should now be running successfully!

For additional support, refer to the main [README.md](README.md) or check the demonstration video for visual setup guidance.