// Example of updating your existing components with the Diagnogenie Design System
import React, { useState } from 'react';
import { 
  Button, 
  Heading1, 
  Heading2, 
  BodyText, 
  ChatBubble, 
  ImageUpload, 
  Alert, 
  Container, 
  Card, 
  LanguageToggle,
  LoadingSpinner 
} from './DesignSystem';

// Enhanced Home Component with Design System
export const EnhancedHome = () => {
  const [language, setLanguage] = useState('en');
  const [showAlert, setShowAlert] = useState(true);

  const content = {
    en: {
      title: "Welcome to Diagnogenie",
      subtitle: "Your Trusted Medical Companion",
      description: "Get accurate medical insights through AI-powered diagnosis and consultation.",
      startChat: "Start Medical Consultation",
      uploadImage: "Upload Medical Image",
      learnMore: "Learn More"
    },
    hi: {
      title: "डायग्नोजीनी में आपका स्वागत है",
      subtitle: "आपका विश्वसनीय चिकित्सा साथी",
      description: "एआई-संचालित निदान और परामर्श के माध्यम से सटीक चिकित्सा अंतर्दृष्टि प्राप्त करें।",
      startChat: "चिकित्सा परामर्श शुरू करें",
      uploadImage: "चिकित्सा छवि अपलोड करें",
      learnMore: "और जानें"
    }
  };

  const currentContent = content[language];

  return (
    <Container>
      <LanguageToggle 
        currentLanguage={language}
        onLanguageChange={setLanguage}
      />

      {showAlert && (
        <Alert 
          type="info" 
          className="mt-l"
          onClose={() => setShowAlert(false)}
        >
          {language === 'en' 
            ? "This is a medical assistance tool. Always consult with healthcare professionals for serious concerns."
            : "यह एक चिकित्सा सहायता उपकरण है। गंभीर चिंताओं के लिए हमेशा स्वास्थ्य पेशेवरों से सलाह लें।"
          }
        </Alert>
      )}

      <div className="text-center py-xxl">
        <Heading1 className="mb-m">{currentContent.title}</Heading1>
        <Heading2 className="mb-l text-primary">{currentContent.subtitle}</Heading2>
        <BodyText className="mb-xl max-w-2xl mx-auto">
          {currentContent.description}
        </BodyText>

        <div className="flex flex-col tablet:flex-row gap-m justify-center items-center mb-xxl">
          <Button onClick={() => console.log('Start chat')}>
            {currentContent.startChat}
          </Button>
          <Button variant="secondary" onClick={() => console.log('Upload image')}>
            {currentContent.uploadImage}
          </Button>
          <Button variant="secondary" onClick={() => console.log('Learn more')}>
            {currentContent.learnMore}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 tablet:grid-cols-2 desktop:grid-cols-3 gap-l mb-xxl">
        <Card>
          <div className="text-center">
            <div className="text-4xl mb-m">🩺</div>
            <Heading3 className="mb-s">
              {language === 'en' ? 'AI Diagnosis' : 'एआई निदान'}
            </Heading3>
            <BodyText variant="secondary">
              {language === 'en' 
                ? 'Advanced AI technology for preliminary medical analysis'
                : 'प्रारंभिक चिकित्सा विश्लेषण के लिए उन्नत एआई तकनीक'
              }
            </BodyText>
          </div>
        </Card>

        <Card>
          <div className="text-center">
            <div className="text-4xl mb-m">💬</div>
            <Heading3 className="mb-s">
              {language === 'en' ? 'Chat Support' : 'चैट सहायता'}
            </Heading3>
            <BodyText variant="secondary">
              {language === 'en' 
                ? 'Interactive consultation with medical AI assistant'
                : 'मेडिकल एआई असिस्टेंट के साथ इंटरैक्टिव परामर्श'
              }
            </BodyText>
          </div>
        </Card>

        <Card>
          <div className="text-center">
            <div className="text-4xl mb-m">📸</div>
            <Heading3 className="mb-s">
              {language === 'en' ? 'Image Analysis' : 'छवि विश्लेषण'}
            </Heading3>
            <BodyText variant="secondary">
              {language === 'en' 
                ? 'Upload medical images for AI-powered analysis'
                : 'एआई-संचालित विश्लेषण के लिए चिकित्सा छवियां अपलोड करें'
              }
            </BodyText>
          </div>
        </Card>
      </div>
    </Container>
  );
};

// Enhanced Chatbot Component with Design System
export const EnhancedChatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Hello! I\'m your medical AI assistant. How can I help you today?',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [language, setLanguage] = useState('en');

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simulate API call
    setTimeout(() => {
      const botResponse = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Thank you for sharing your symptoms. Based on what you\'ve described, I recommend consulting with a healthcare professional for a proper evaluation.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botResponse]);
      setIsLoading(false);
    }, 2000);
  };

  return (
    <Container className="max-w-4xl">
      <LanguageToggle 
        currentLanguage={language}
        onLanguageChange={setLanguage}
      />

      <Card className="h-96 tablet:h-[500px] overflow-y-auto mb-m">
        <div className="p-m">
          {messages.map(message => (
            <ChatBubble 
              key={message.id} 
              type={message.type}
              className="mb-s"
            >
              {message.content}
            </ChatBubble>
          ))}
          {isLoading && (
            <div className="flex items-center gap-s">
              <LoadingSpinner size="sm" />
              <BodyText variant="secondary">AI is thinking...</BodyText>
            </div>
          )}
        </div>
      </Card>

      <div className="flex gap-s">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder={language === 'en' ? 'Describe your symptoms...' : 'अपने लक्षणों का वर्णन करें...'}
          className="flex-1 px-m py-s border border-neutral-light-gray rounded-button focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
        />
        <Button onClick={handleSendMessage} disabled={isLoading}>
          {language === 'en' ? 'Send' : 'भेजें'}
        </Button>
      </div>
    </Container>
  );
};

// Enhanced Image Upload Component with Design System
export const EnhancedImageUpload = () => {
  const [uploadStatus, setUploadStatus] = useState('default');
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [language, setLanguage] = useState('en');

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setUploadStatus('success');
    
    // Simulate image analysis
    setTimeout(() => {
      setAnalysisResult({
        confidence: '85%',
        findings: language === 'en' 
          ? 'Preliminary analysis suggests consulting with a radiologist for detailed evaluation.'
          : 'प्रारंभिक विश्लेषण विस्तृत मूल्यांकन के लिए रेडियोलॉजिस्ट से परामर्श का सुझाव देता है।'
      });
    }, 3000);
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      setAnalysisResult(null);
      setUploadStatus('default');
      // Trigger analysis
      setTimeout(() => {
        setAnalysisResult({
          confidence: '92%',
          findings: language === 'en'
            ? 'AI analysis complete. Please consult with a medical professional for interpretation.'
            : 'एआई विश्लेषण पूर्ण। व्याख्या के लिए चिकित्सा पेशेवर से सलाह लें।'
        });
      }, 2000);
    }
  };

  return (
    <Container className="max-w-2xl">
      <LanguageToggle 
        currentLanguage={language}
        onLanguageChange={setLanguage}
      />

      <Card className="text-center">
        <Heading2 className="mb-l">
          {language === 'en' ? 'Medical Image Analysis' : 'चिकित्सा छवि विश्लेषण'}
        </Heading2>

        <ImageUpload
          onFileSelect={handleFileSelect}
          status={uploadStatus}
          className="mb-l"
        />

        {selectedFile && (
          <Alert type="success" className="mb-l">
            {language === 'en' 
              ? `File uploaded: ${selectedFile.name}`
              : `फ़ाइल अपलोड की गई: ${selectedFile.name}`
            }
          </Alert>
        )}

        {selectedFile && (
          <Button onClick={handleAnalyze} className="mb-l">
            {language === 'en' ? 'Analyze Image' : 'छवि का विश्लेषण करें'}
          </Button>
        )}

        {analysisResult && (
          <Card className="bg-primary-light">
            <Heading3 className="mb-m">
              {language === 'en' ? 'Analysis Results' : 'विश्लेषण परिणाम'}
            </Heading3>
            <BodyText className="mb-s">
              <strong>
                {language === 'en' ? 'Confidence:' : 'विश्वास:'} {analysisResult.confidence}
              </strong>
            </BodyText>
            <BodyText>{analysisResult.findings}</BodyText>
          </Card>
        )}
      </Card>
    </Container>
  );
};

export { EnhancedHome, EnhancedChatbot, EnhancedImageUpload };