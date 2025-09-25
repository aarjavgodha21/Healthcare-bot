import React, { useState, useRef, useEffect } from "react";
import { 
  Button, 
  Heading2, 
  BodyText, 
  Alert, 
  ChatBubble, 
  Container, 
  LoadingSpinner 
} from './components/DesignSystem';

export default function Chatbot({ onNavigate, language = 'en' }) {
  const [messages, setMessages] = useState([
    { 
      role: "assistant", 
      content: language === 'en' 
        ? "Hello! I'm your AI medical assistant. I can help analyze symptoms, provide medical guidance, and answer health-related questions. How can I assist you today?"
        : "नमस्ते! मैं आपका AI चिकित्सा सहायक हूं। मैं लक्षणों का विश्लेषण करने, चिकित्सा मार्गदर्शन प्रदान करने और स्वास्थ्य संबंधी प्रश्नों के उत्तर देने में मदद कर सकता हूं। मैं आपकी कैसे सहायता कर सकता हूं?"
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const recognitionRef = useRef(null);
  const synthRef = useRef(null);

  // Content translations
  const content = {
    en: {
      title: "AI Medical Consultation",
      subtitle: "Advanced Clinical Decision Support System",
      aiOnline: "AI Online",
      accuracy: "95% Accuracy",
      clinicalGrade: "Clinical Grade AI",
      patientQuery: "Patient Query",
      analyzingSymptoms: "AI is analyzing your symptoms...",
      connectionError: "Connection Error",
      placeholder: "Describe your symptoms, ask about medications, or request medical guidance...",
      stopRecording: "Stop recording", 
      voiceInput: "Voice input (Auto-detect Hindi/English)",
      send: "Send",
      analyzing: "Analyzing",
      listenHindi: "🇮🇳 Listen in Hindi",
      listenEnglish: "🇺🇸 Listen in English",
      stopSpeaking: "Stop speaking",
      quickTemplates: {
        fever: "I have a persistent fever of 101.6°F with chills and fatigue",
        chest: "I'm experiencing chest pain and shortness of breath", 
        headache: "I have a severe headache with nausea and light sensitivity",
        drugInfo: "What are the side effects of ibuprofen and acetaminophen?"
      }
    },
    hi: {
      title: "AI चिकित्सा परामर्श",
      subtitle: "उन्नत नैदानिक निर्णय सहायता प्रणाली",
      aiOnline: "AI ऑनलाइन",
      accuracy: "95% सटीकता",
      clinicalGrade: "क्लिनिकल ग्रेड AI",
      patientQuery: "रोगी प्रश्न",
      analyzingSymptoms: "AI आपके लक्षणों का विश्लेषण कर रहा है...",
      connectionError: "कनेक्शन त्रुटि",
      placeholder: "अपने लक्षणों का वर्णन करें, दवाओं के बारे में पूछें, या चिकित्सा मार्गदर्शन का अनुरोध करें...",
      stopRecording: "रिकॉर्डिंग बंद करें",
      voiceInput: "आवाज इनपुट (हिंदी/अंग्रेजी अपने आप पहचान)",
      send: "भेजें",
      analyzing: "विश्लेषण कर रहे हैं",
      listenHindi: "🇮🇳 हिंदी में सुनें",
      listenEnglish: "🇺🇸 अंग्रेजी में सुनें", 
      stopSpeaking: "बोलना बंद करें",
      quickTemplates: {
        fever: "मुझे 101.6°F बुखार है साथ में ठंड लग रही है और थकान है",
        chest: "मुझे सीने में दर्द और सांस लेने में तकलीफ हो रही है",
        headache: "मुझे तेज सिरदर्द है साथ में उल्टी और रोशनी से परेशानी हो रही है", 
        drugInfo: "इबुप्रोफेन और एसिटामिनोफेन के दुष्प्रभाव क्या हैं?"
      }
    }
  };

  const t = content[language] || content.en;

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    setInput("");
    
    try {
      const res = await fetch("http://127.0.0.1:8001/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          messages: newMessages,
          language: language // Pass language preference to backend
        }),
      });
      if (!res.ok) throw new Error("Network error");
      const data = await res.json();
      console.log('Backend response:', data.reply); // Debug log
      
      const assistantMessage = { role: "assistant", content: data.reply };
      setMessages([...newMessages, assistantMessage]);
      
    } catch (err) {
      const errorMsg = language === 'en' 
        ? "Failed to get reply. Please check if the backend server is running."
        : "उत्तर प्राप्त करने में विफल। कृपया जांचें कि बैकएंड सर्वर चल रहा है।";
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  // Initialize speech recognition and synthesis
  useEffect(() => {
    // Speech Recognition Setup
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      
      // Set language based on app language preference
      recognitionRef.current.lang = language === 'hi' ? 'hi-IN' : 'en-US';
      
      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
      };
      
      recognitionRef.current.onerror = (event) => {
        setIsListening(false);
        console.error('Speech recognition error:', event.error);
        const errorMsg = language === 'en' 
          ? `Voice recognition error: ${event.error}. Please try again.`
          : `आवाज पहचान त्रुटि: ${event.error}। कृपया पुनः प्रयास करें।`;
        setError(errorMsg);
      };
      
      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
    
    // Speech Synthesis Setup
    if ('speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis;
      // Load voices
      const loadVoices = () => {
        const voices = synthRef.current.getVoices();
        console.log('Available voices:', voices.map(v => `${v.name} (${v.lang})`));
      };
      synthRef.current.onvoiceschanged = loadVoices;
      loadVoices();
    }
  }, [language]); // Re-initialize when language changes

  // Check if text contains Hindi characters
  const containsHindi = (text) => {
    const hindiRegex = /[\u0900-\u097F]/;
    return hindiRegex.test(text);
  };

  // Format medical diagnosis text for beautiful display
  const formatDiagnosisText = (text) => {
    if (!text) return text;
    
    // Split into lines and process each section
    const lines = text.split('\n').filter(line => line.trim());
    const sections = [];
    let currentSection = null;
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      
      // Detect main headers
      if (trimmed.includes('MEDICAL CONSULTATION REPORT')) {
        sections.push({ type: 'main-header', text: '🏥 Medical Consultation Report' });
      }
      else if (trimmed.includes('CHIEF COMPLAINT')) {
        if (currentSection) sections.push(currentSection);
        currentSection = { type: 'section', title: '📋 Chief Complaint', items: [] };
      }
      else if (trimmed.includes('GENERAL ASSESSMENT')) {
        if (currentSection) sections.push(currentSection);
        currentSection = { type: 'section', title: '🔍 General Assessment', items: [] };
      }
      else if (trimmed.includes('GENERAL RECOMMENDATIONS')) {
        if (currentSection) sections.push(currentSection);
        currentSection = { type: 'section', title: '💊 General Recommendations', items: [] };
      }
      else if (trimmed.includes('RED FLAGS')) {
        if (currentSection) sections.push(currentSection);
        currentSection = { type: 'section', title: '⚠️ Red Flags - Seek Immediate Care If:', items: [] };
      }
      else if (trimmed.includes('FOLLOW-UP')) {
        if (currentSection) sections.push(currentSection);
        currentSection = { type: 'section', title: '📅 Follow-Up', items: [] };
      }
      else if (trimmed.includes('MEDICAL DISCLAIMER')) {
        if (currentSection) sections.push(currentSection);
        currentSection = { type: 'section', title: '⚕️ Medical Disclaimer', items: [] };
      }
      // Process content lines
      else if (trimmed.startsWith('Primary Concern:')) {
        if (currentSection) {
          currentSection.items.push({ type: 'highlight', text: trimmed.replace('Primary Concern:', '').trim() });
        }
      }
      else if (trimmed.startsWith('•') || trimmed.startsWith('-')) {
        if (currentSection) {
          currentSection.items.push({ type: 'bullet', text: trimmed.replace(/^[•\-]\s*/, '') });
        }
      }
      else if (trimmed.length > 0) {
        if (currentSection) {
          currentSection.items.push({ type: 'text', text: trimmed });
        }
      }
    }
    
    if (currentSection) sections.push(currentSection);
    return sections;
  };

  // Start voice recording
  const startListening = () => {
    if (!recognitionRef.current) {
      const errorMsg = language === 'en' 
        ? "Speech recognition not supported in this browser."
        : "इस ब्राउज़र में आवाज पहचान समर्थित नहीं है।";
      setError(errorMsg);
      return;
    }
    
    setError("");
    setIsListening(true);
    
    // Set language for speech recognition
    recognitionRef.current.lang = language === 'hi' ? 'hi-IN' : 'en-US';
    
    try {
      recognitionRef.current.start();
    } catch (err) {
      setIsListening(false);
      const errorMsg = language === 'en' 
        ? "Could not start voice recognition. Please try again."
        : "आवाज पहचान शुरू नहीं हो सका। कृपया पुनः प्रयास करें।";
      setError(errorMsg);
    }
  };

  // Stop voice recording
  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  // Text to speech function with proper language support
  const speakText = (text, ttsLanguage = null) => {
    // Check if speech synthesis is supported
    if (!('speechSynthesis' in window) || !synthRef.current) {
      const errorMsg = language === 'en' 
        ? "Speech synthesis not supported in this browser."
        : "इस ब्राउज़र में आवाज संश्लेषण समर्थित नहीं है।";
      setError(errorMsg);
      return;
    }
    
    // Stop any current speech
    try {
      if (synthRef.current.speaking) {
        synthRef.current.cancel();
      }
    } catch (cancelError) {
      console.warn('Error cancelling speech:', cancelError);
    }
    
    // Determine language for TTS
    const speechLang = ttsLanguage || (language === 'hi' ? 'hi-IN' : 'en-US');
    
    // Extract key medical information for summary
    const medicalSummary = extractMedicalSummary(text);
    console.log('Speaking text:', medicalSummary);
    console.log('Language:', speechLang);
    
    const utterance = new SpeechSynthesisUtterance(medicalSummary);
    
    // Wait for voices to load
    const speak = () => {
      const voices = synthRef.current.getVoices();
      console.log('Available voices:', voices.length);
      
      let selectedVoice = null;
      
      if (speechLang === 'hi-IN') {
        // Look for Hindi voices more thoroughly
        selectedVoice = voices.find(voice => 
          voice.lang === 'hi-IN' || 
          voice.lang === 'hi' ||
          voice.lang.startsWith('hi-') ||
          voice.name.toLowerCase().includes('hindi') ||
          voice.name.toLowerCase().includes('हिन्दी')
        );
        
        console.log('Hindi voice found:', selectedVoice?.name);
        
        if (!selectedVoice) {
          // Try Google voices or other alternatives
          selectedVoice = voices.find(voice => 
            voice.name.toLowerCase().includes('google') && 
            (voice.lang.includes('hi') || voice.name.toLowerCase().includes('hindi'))
          );
        }
        
        utterance.lang = 'hi-IN';
      } else {
        // English voice
        selectedVoice = voices.find(voice => 
          voice.lang === 'en-US' || 
          voice.lang.startsWith('en-') ||
          voice.lang === 'en'
        );
        utterance.lang = 'en-US';
      }
      
      if (selectedVoice) {
        utterance.voice = selectedVoice;
        console.log('Using voice:', selectedVoice.name, selectedVoice.lang);
      } else {
        console.log('No specific voice found, using default');
      }
      
      utterance.rate = 0.8;
      utterance.pitch = 1;
      utterance.volume = 1;
      
      utterance.onstart = () => {
        setIsSpeaking(true);
        setError("");
      };
      
      utterance.onend = () => {
        setIsSpeaking(false);
      };
      
      utterance.onerror = (event) => {
        setIsSpeaking(false);
        console.error('Speech synthesis error:', event.error);
        
        // Don't show error for interruption (user stopped speech)
        if (event.error === 'interrupted' || event.error === 'canceled') {
          console.log('Speech was interrupted by user');
          return;
        }
        
        // Only show error for actual problems
        const errorMsg = language === 'en' 
          ? `Speech error: ${event.error}. ${speechLang === 'hi-IN' ? 'Hindi voice may not be available.' : 'Try selecting a different language.'}`
          : `आवाज त्रुटि: ${event.error}. ${speechLang === 'hi-IN' ? 'हिंदी आवाज उपलब्ध नहीं हो सकती।' : 'अन्य भाषा चुनने का प्रयास करें।'}`;
        setError(errorMsg);
      };
      
      try {
        synthRef.current.speak(utterance);
      } catch (err) {
        setIsSpeaking(false);
        const errorMsg = language === 'en' 
          ? "Could not start speech synthesis. Please try again."
          : "आवाज संश्लेषण शुरू नहीं हो सका। कृपया पुनः प्रयास करें।";
        setError(errorMsg);
      }
    };
    
    // If voices aren't loaded yet, wait for them
    if (synthRef.current.getVoices().length === 0) {
      synthRef.current.onvoiceschanged = () => {
        speak();
      };
    } else {
      speak();
    }
  };

  // Extract medical summary from text with better diagnosis focus
  const extractMedicalSummary = (text) => {
    if (!text) return "";
    
    // First, clean the text by removing markdown
    const cleanText = text
      .replace(/#{1,6}\s*/g, '') // Remove # headers
      .replace(/\*\*([^*]+)\*\*/g, '$1') // Remove **bold**
      .replace(/\*([^*]+)\*/g, '$1') // Remove *italic*
      .replace(/[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]/gu, ''); // Remove emojis
    
    const lines = cleanText.split('\n').filter(line => line.trim());
    let summary = "";
    let keyFindings = [];
    
    // Enhanced medical keywords for both languages
    const criticalKeywords = [
      'PRIMARY CONCERN', 'CHIEF COMPLAINT', 'MAIN SYMPTOM', 'DIAGNOSIS',
      'मुख्य चिंता', 'मुख्य शिकायत', 'प्राथमिक लक्षण', 'निदान'
    ];
    
    const treatmentKeywords = [
      'TREATMENT', 'MEDICATION', 'PRESCRIPTION', 'RECOMMENDATION', 'ADVICE',
      'उपचार', 'दवा', 'चिकित्सा', 'सिफारिश', 'सलाह'
    ];
    
    // Extract the most important medical information
    for (const line of lines) {
      const upperLine = line.toUpperCase();
      const cleanLine = line.replace(/[#*•\-\d\.]/g, '').trim();
      
      if (cleanLine.length < 10) continue; // Skip very short lines
      
      // Prioritize critical findings
      const isCritical = criticalKeywords.some(keyword => 
        upperLine.includes(keyword.toUpperCase())
      );
      
      const isTreatment = treatmentKeywords.some(keyword => 
        upperLine.includes(keyword.toUpperCase())
      );
      
      if (isCritical || isTreatment) {
        keyFindings.push(cleanLine);
      }
      
      // Also capture lines that look like medical statements
      if (cleanLine.match(/(fever|temperature|pain|symptoms?|condition|diagnosis|treatment)/i) ||
          cleanLine.match(/(बुखार|तापमान|दर्द|लक्षण|स्थिति|निदान|उपचार)/)) {
        if (cleanLine.length <= 100) { // Keep it concise
          keyFindings.push(cleanLine);
        }
      }
    }
    
    // Build summary from key findings
    if (keyFindings.length > 0) {
      // Take the most important 3-4 findings
      summary = keyFindings.slice(0, 4).join('. ') + '.';
    } else {
      // Fallback: extract first meaningful sentences
      const sentences = cleanText.split(/[.!?।]/).filter(s => 
        s.trim().length > 15 && s.trim().length < 80
      );
      summary = sentences.slice(0, 3).join('. ') + '.';
    }
    
    // Final cleanup and length check
    summary = summary
      .replace(/\s+/g, ' ') // Clean up spaces
      .replace(/\.\s*\./g, '.') // Remove double periods
      .trim();
    
    // Ensure reasonable length for speech
    if (summary.length > 200) {
      summary = summary.substring(0, 197) + "...";
    }
    
    // Fallback if still no good summary
    if (summary.length < 20) {
      summary = language === 'en' 
        ? "Medical consultation completed. Please review the detailed diagnosis and recommendations provided."
        : "चिकित्सा परामर्श पूरा हुआ। कृपया प्रदान किए गए विस्तृत निदान और सिफारिशों की समीक्षा करें।";
    }
    
    return summary;
  };

  // Stop speaking
  const stopSpeaking = () => {
    try {
      if (synthRef.current) {
        if (synthRef.current.speaking) {
          synthRef.current.cancel();
        }
        setIsSpeaking(false);
        // Clear any speech-related errors when user manually stops
        setError("");
      } else {
        console.log('Speech synthesis not available');
        setIsSpeaking(false);
      }
    } catch (error) {
      console.error('Speech synthesis stop error:', error);
      setIsSpeaking(false);
      // Clear any error states
      setError("");
    }
  };

  // Update initial message when language changes
  useEffect(() => {
    if (messages.length === 1 && messages[0].role === "assistant") {
      const newMessage = {
        role: "assistant",
        content: language === 'en' 
          ? "Hello! I'm your AI medical assistant. I can help analyze symptoms, provide medical guidance, and answer health-related questions. How can I assist you today?"
          : "नमस्ते! मैं आपका AI चिकित्सा सहायक हूं। मैं लक्षणों का विश्लेषण करने, चिकित्सा मार्गदर्शन प्रदान करने और स्वास्थ्य संबंधी प्रश्नों के उत्तर देने में मदद कर सकता हूं। मैं आपकी कैसे सहायता कर सकता हूं?"
      };
      setMessages([newMessage]);
    }
  }, [language]);

  return (
    <Container className="min-h-screen bg-gradient-to-br from-neutral-off-white via-primary-light to-primary-light/50">
      {/* Header Section */}
      <div className="bg-white/90 backdrop-blur-lg border-b border-primary/20 shadow-lg">
        <div className="max-w-6xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant="primary"
                onClick={() => onNavigate('home')}
                className="group w-12 h-12 bg-gradient-to-br from-secondary-success to-secondary-success/80 rounded-2xl flex items-center justify-center hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-primary/20"
              >
                <svg className="w-6 h-6 text-white group-hover:-translate-x-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/>
                </svg>
              </Button>
              <div>
                <Heading2 className="text-2xl font-bold bg-gradient-to-r from-secondary-success to-secondary-success/80 bg-clip-text text-transparent">
                  {t.title}
                </Heading2>
                <BodyText variant="small" className="text-neutral-medium-gray">{t.subtitle}</BodyText>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 bg-secondary-success/10 px-4 py-2 rounded-full border border-secondary-success/20">
                <div className="w-2 h-2 bg-secondary-success rounded-full animate-pulse"></div>
                <BodyText variant="small" className="text-secondary-success font-medium">{t.aiOnline}</BodyText>
              </div>
              <div className="text-right">
                <BodyText className="font-semibold text-neutral-dark-gray">{t.accuracy}</BodyText>
                <BodyText variant="small" className="text-neutral-medium-gray">{t.clinicalGrade}</BodyText>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Chat Interface */}
      <div className="max-w-6xl mx-auto p-6 mt-8">
        <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl border border-primary/20 h-[600px] flex flex-col overflow-hidden">
          
          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto p-8 space-y-4" style={{fontFamily: language === 'hi' ? 'var(--font-hindi)' : 'var(--font-primary)'}}>
            {messages.map((msg, i) => (
              <ChatBubble
                key={i}
                role={msg.role}
                content={msg.content}
                formattedSections={msg.role === "assistant" ? formatDiagnosisText(msg.content) : null}
                language={language}
                timestamp="Just now"
                onSpeak={(ttsLang) => speakText(msg.content, ttsLang)}
                isSpeaking={isSpeaking}
                onStopSpeak={stopSpeaking}
                showLanguageToggle={msg.role === "assistant"}
              />
            ))}
            
            {/* Loading Indicator */}
            {loading && (
              <div className="flex justify-start">
                <div className="max-w-4xl flex flex-row items-start space-x-4">
                  <div className="w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0 shadow-lg bg-gradient-to-br from-secondary-success to-secondary-success/80 mr-4">
                    <svg className="w-6 h-6 text-white animate-pulse" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                    </svg>
                  </div>
                  <div className="relative px-6 py-4 rounded-2xl shadow-lg bg-white border border-neutral-light-gray">
                    <div className="flex items-center space-x-3">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-secondary-success rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-secondary-success rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-secondary-success rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                      <BodyText variant="small" className="text-neutral-medium-gray">{t.analyzingSymptoms}</BodyText>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="px-6 pb-3">
              <Alert variant="error" className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-secondary-error/20 rounded-full flex items-center justify-center">
                  <svg className="w-4 h-4 text-secondary-error" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                  </svg>
                </div>
                <div>
                  <BodyText className="font-medium text-secondary-error">{t.connectionError}</BodyText>
                  <BodyText variant="small" className="text-secondary-error">{error}</BodyText>
                </div>
              </Alert>
            </div>
          )}

          {/* Input Section */}
          <div className="p-8 border-t border-neutral-light-gray bg-neutral-off-white/50">
            <form onSubmit={sendMessage} className="space-y-4">
              <div className="flex space-x-4">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    className={`w-full px-6 py-4 pr-14 rounded-2xl border border-neutral-light-gray focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent text-neutral-dark-gray placeholder-neutral-medium-gray bg-white shadow-lg ${
                      language === 'hi' ? 'font-hindi' : 'font-primary'
                    }`}
                    placeholder={t.placeholder}
                    disabled={loading}
                    lang={language === 'hi' ? 'hi' : 'en'}
                  />
                  {/* Voice Input Button */}
                  <div className="absolute right-2 top-2">
                    <button
                      type="button"
                      onClick={() => isListening ? stopListening() : startListening()}
                      className={`p-2 rounded-xl transition-all ${
                        isListening 
                          ? 'bg-secondary-error text-white animate-pulse' 
                          : 'bg-primary-light text-primary hover:bg-primary/10'
                      }`}
                      disabled={loading}
                      title={isListening ? t.stopRecording : t.voiceInput}
                    >
                      {isListening ? (
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M6 6h12v12H6z"/>
                        </svg>
                      ) : (
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z"/>
                        </svg>
                      )}
                    </button>
                  </div>
                </div>
                <Button
                  type="submit"
                  variant="primary"
                  className="px-8 py-4 bg-gradient-to-r from-primary to-primary-dark text-white font-semibold rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105"
                  disabled={loading || !input.trim()}
                >
                  {loading ? (
                    <div className="flex items-center space-x-2">
                      <LoadingSpinner size="xs" color="white" />
                      <span>{t.analyzing}</span>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-2">
                      <span>{t.send}</span>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
                      </svg>
                    </div>
                  )}
                </Button>
              </div>
              
              {/* Quick Consultation Templates */}
              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    setInput(t.quickTemplates.fever);
                  }}
                  className="px-4 py-2 bg-primary-light/50 text-primary hover:bg-primary-light border border-primary/20 rounded-xl text-sm flex items-center space-x-2 transition-colors"
                >
                  <span>🌡️</span>
                  <span>{language === 'en' ? 'Fever & Chills' : 'बुखार और ठंड'}</span>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    setInput(t.quickTemplates.chest);
                  }}
                  className="px-4 py-2 bg-secondary-error/10 text-secondary-error hover:bg-secondary-error/20 border border-secondary-error/20 rounded-xl text-sm flex items-center space-x-2 transition-colors"
                >
                  <span>🫁</span>
                  <span>{language === 'en' ? 'Chest Pain' : 'सीने में दर्द'}</span>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    setInput(t.quickTemplates.headache);
                  }}
                  className="px-4 py-2 bg-purple-50 text-purple-700 hover:bg-purple-100 border border-purple-200 rounded-xl text-sm flex items-center space-x-2 transition-colors"
                >
                  <span>🧠</span>
                  <span>{language === 'en' ? 'Migraine' : 'सिरदर्द'}</span>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    setInput(t.quickTemplates.drugInfo);
                  }}
                  className="px-4 py-2 bg-secondary-success/10 text-secondary-success hover:bg-secondary-success/20 border border-secondary-success/20 rounded-xl text-sm flex items-center space-x-2 transition-colors"
                >
                  <span>💊</span>
                  <span>{language === 'en' ? 'Drug Information' : 'दवा की जानकारी'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </Container>
  );
}
