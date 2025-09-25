import { useState, useEffect } from 'react'
import Home from './Home'
import Chatbot from './Chatbot'
import ImageUpload from './ImageUpload'
import { LanguageToggle } from './components/DesignSystem'
import './index.css'

function App() {
  const [currentView, setCurrentView] = useState('home')
  const [language, setLanguage] = useState('en')

  // Always show new views from the top
  useEffect(() => {
    // Reset the window scroll position on view change
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' })
  }, [currentView])

  const renderView = () => {
    switch (currentView) {
      case 'chatbot':
        return <Chatbot onNavigate={setCurrentView} language={language} />
      case 'imageUpload':
        return <ImageUpload onNavigate={setCurrentView} language={language} />
      default:
        return <Home onNavigate={setCurrentView} language={language} />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-off-white via-primary-light to-primary-light/50">
      {/* Language Toggle - Fixed Position */}
      <LanguageToggle 
        currentLanguage={language}
        onLanguageChange={setLanguage}
      />

      {/* Medical Header */}
      <header className="bg-gradient-to-r from-primary-dark via-primary to-primary shadow-xl border-b-4 border-primary-light">
        <div className="container mx-auto px-l py-m">
          <div className="flex items-center justify-between pr-24">
            <div className="flex items-center space-x-m flex-1">
              <div className="bg-white/20 backdrop-blur-sm rounded-card p-s">
                {/* Healthcare icon: more detailed stethoscope (white stroke) */}
                <svg className="w-8 h-8" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <g fill="none" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    {/* Ear tips */}
                    <circle cx="6" cy="4" r="1" />
                    <circle cx="14" cy="4" r="1" />
                    {/* Upper tubes */}
                    <path d="M6 5v3a4 4 0 0 0 8 0V5" />
                    {/* Flexible Y-tube to chest piece */}
                    <path d="M10 12v2a4 4 0 0 0 8 0v-1" />
                    {/* Chest piece ring and body */}
                    <circle cx="19" cy="16" r="2.2" />
                    <circle cx="19" cy="16" r="1" />
                    {/* Short connector stem */}
                    <path d="M18 14.5v-1" />
                  </g>
                </svg>
              </div>
              <div className="flex-1">
                <h1 className="text-h2 font-bold text-white tracking-tight">
                  {language === 'en' ? 'DiagnoGenie' : 'डायग्नोजीनी'}
                </h1>
                <p className="text-primary-light text-body-small font-medium">
                  {language === 'en' 
                    ? 'AI-Powered Medical Imaging & Consultation'
                    : 'एआई-संचालित चिकित्सा इमेजिंग और परामर्श'
                  }
                </p>
              </div>
            </div>
            
            {currentView !== 'home' && (
              <button
                onClick={() => setCurrentView('home')}
                className="btn-secondary bg-white/20 hover:bg-white/30 backdrop-blur-sm text-white border-white/30 hover:border-white/50 flex items-center space-x-s transition-all duration-200 hover:scale-105 relative z-10"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/>
                </svg>
                <span className="font-medium">
                  {language === 'en' ? 'Home' : 'होم'}
                </span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-l py-xl">
        {renderView()}
      </main>

      {/* Medical Footer */}
      <footer className="bg-neutral-dark-gray text-neutral-medium-gray py-xl mt-xxl">
        <div className="container mx-auto px-l">
          <div className="grid md:grid-cols-3 gap-xl">
            <div>
              <h3 className="text-white font-semibold mb-m flex items-center">
                <svg className="w-5 h-5 mr-s text-primary" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
                {language === 'en' ? 'Medical AI Platform' : 'चिकित्सा एआई प्लेटफॉर्म'}
              </h3>
              <p className="text-body-small leading-relaxed">
                {language === 'en' 
                  ? 'Advanced AI-powered medical imaging analysis and consultation platform designed to assist healthcare professionals with diagnostic support.'
                  : 'उन्नत एआई-संचालित चिकित्सा इमेजिंग विश्लेषण और परामर्श प्लेटफॉर्म जो स्वास्थ्य पेशेवरों की निदान सहायता के लिए डिज़ाइन किया गया है।'
                }
              </p>
            </div>
            
            <div>
              <h3 className="text-white font-semibold mb-m flex items-center">
                <svg className="w-5 h-5 mr-s text-secondary-success" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                {language === 'en' ? 'Safety & Compliance' : 'सुरक्षा और अनुपालन'}
              </h3>
              <ul className="text-body-small space-y-s">
                <li>• {language === 'en' ? 'HIPAA-compliant processing' : 'HIPAA-अनुपालित प्रसंस्करण'}</li>
                <li>• {language === 'en' ? 'Medical-grade AI models' : 'चिकित्सा-ग्रेड एआई मॉडल'}</li>
                <li>• {language === 'en' ? 'Professional oversight required' : 'पेशेवर निरीक्षण आवश्यक'}</li>
                <li>• {language === 'en' ? 'Educational use only' : 'केवल शैक्षिक उपयोग'}</li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-white font-semibold mb-m flex items-center">
                <svg className="w-5 h-5 mr-s text-secondary-warning" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"/>
                </svg>
                {language === 'en' ? 'Important Disclaimer' : 'महत्वपूर्ण अस्वीकरण'}
              </h3>
              <p className="text-body-small leading-relaxed">
                {language === 'en' 
                  ? 'This AI system provides educational support only. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.'
                  : 'यह एआई सिस्टम केवल शैक्षिक सहायता प्रदान करता है। चिकित्सा निदान और उपचार निर्णयों के लिए हमेशा योग्य स्वास्थ्य पेशेवरों से सलाह लें।'
                }
              </p>
            </div>
          </div>
          
          <div className="border-t border-neutral-medium-gray mt-xl pt-l text-center text-body-small">
            <p>&copy; 2025 {language === 'en' ? 'DiagnoGenie' : 'डायग्नोजीनी'}. {language === 'en' ? 'AI-Powered Medical Platform. For educational and research purposes.' : 'एआई-संचालित चिकित्सा प्लेटफॉर्म। शैक्षिक और अनुसंधान उद्देश्यों के लिए।'}</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
