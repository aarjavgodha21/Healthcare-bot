import React, { useState } from "react";
import { 
  Heading1, 
  Heading2, 
  Heading3, 
  BodyText, 
  Card, 
  Container, 
  Alert,
  Button,
  Spacer 
} from './components/DesignSystem';

export default function Home({ onNavigate, language = 'en' }) {
  const [alertVisible, setAlertVisible] = useState(true);

  const content = {
    en: {
      alert: "This is a medical assistance tool. Always consult with healthcare professionals for serious concerns.",
      heroTitle: "Advanced Medical AI Diagnosis",
      heroSubtitle: "Your Trusted Medical Companion",
      heroDescription: "Harness the power of cutting-edge artificial intelligence for comprehensive medical imaging analysis and intelligent clinical consultation support",
      features: {
        fdaGrade: "FDA-Grade AI Models",
        realtime: "Real-time Analysis", 
        hipaa: "HIPAA Compliant",
        accuracy: "95%+ Accuracy"
      },
      imageAnalysis: {
        title: "Medical Imaging Analysis",
        subtitle: "X-ray & MRI Analysis",
        description: "Upload X-rays, CT scans, or MRI images for comprehensive AI-powered analysis with advanced visualization techniques and diagnostic insights.",
        features: [
          "18 pathology classifications",
          "5 advanced visualization types",
          "Instant confidence scoring",
          "Clinical recommendations"
        ],
        cta: "Start Analysis"
      },
      consultation: {
        title: "Intelligent Medical Consultation", 
        subtitle: "AI Consultation",
        description: "Engage with our advanced medical AI for symptom analysis, differential diagnosis, and evidence-based treatment recommendations.",
        features: [
          "Natural language processing",
          "Differential diagnosis support",
          "Treatment protocols", 
          "Red flag identification"
        ],
        cta: "Start Consultation"
      },
      technology: {
        title: "Powered by Advanced AI Technology",
        description: "Our platform leverages state-of-the-art machine learning models trained on millions of medical images and clinical datasets for unparalleled diagnostic accuracy.",
        features: [
          { title: "CheXNet Model", desc: "Deep learning model trained on 100,000+ chest X-rays" },
          { title: "Real-time Analysis", desc: "Sub-second processing with live confidence scoring" },
          { title: "Clinical Grade", desc: "FDA-equivalent standards with medical oversight" },
          { title: "95%+ Accuracy", desc: "Validated against expert radiologist interpretations" }
        ]
      }
    },
    hi: {
      alert: "यह एक चिकित्सा सहायता उपकरण है। गंभीर चिंताओं के लिए हमेशा स्वास्थ्य पेशेवरों से सलाह लें।",
      heroTitle: "उन्नत चिकित्सा एआई निदान",
      heroSubtitle: "आपका विश्वसनीय चिकित्सा साथी",
      heroDescription: "व्यापक चिकित्सा इमेजिंग विश्लेषण और बुद्धिमान नैदानिक परामर्श सहायता के लिए अत्याधुनिक कृत्रिम बुद्धिमत्ता की शक्ति का उपयोग करें",
      features: {
        fdaGrade: "FDA-ग्रेड एआई मॉडल",
        realtime: "रियल-टाइम विश्लेषण",
        hipaa: "HIPAA अनुपालित",
        accuracy: "95%+ सटीकता"
      },
      imageAnalysis: {
        title: "चिकित्सा इमेजिंग विश्लेषण",
        subtitle: "एक्स-रे और एमआरआई विश्लेषण",
        description: "उन्नत दृश्यीकरण तकनीकों और नैदानिक अंतर्दृष्टि के साथ व्यापक एआई-संचालित विश्लेषण के लिए एक्स-रे, सीटी स्कैन या एमआरआई छवियां अपलोड करें।",
        features: [
          "18 पैथोलॉजी वर्गीकरण",
          "5 उन्नत दृश्यीकरण प्रकार",
          "तत्काल विश्वास स्कोरिंग",
          "नैदानिक सिफारिशें"
        ],
        cta: "विश्लेषण शुरू करें"
      },
      consultation: {
        title: "बुद्धिमान चिकित्सा परामर्श",
        subtitle: "एआई परामर्श",
        description: "लक्षण विश्लेषण, विभेदक निदान और साक्ष्य-आधारित उपचार सिफारिशों के लिए हमारे उन्नत चिकित्सा एआई के साथ जुड़ें।",
        features: [
          "प्राकृतिक भाषा प्रसंस्करण",
          "विभेदक निदान समर्थन",
          "उपचार प्रोटोकॉल",
          "लाल झंडी की पहचान"
        ],
        cta: "परामर्श शुरू करें"
      },
      technology: {
        title: "उन्नत एआई तकनीक द्वारा संचालित",
        description: "हमारा प्लेटफॉर्म अतुलनीय नैदानिक सटीकता के लिए लाखों चिकित्सा छवियों और नैदानिक डेटासेट पर प्रशिक्षित अत्याधुनिक मशीन लर्निंग मॉडल का लाभ उठाता है।",
        features: [
          { title: "चेक्सनेट मॉडल", desc: "100,000+ छाती एक्स-रे पर प्रशिक्षित गहन शिक्षण मॉडल" },
          { title: "रियल-टाइम विश्लेषण", desc: "लाइव कॉन्फिडेंस स्कोरिंग के साथ सब-सेकेंड प्रसंस्करण" },
          { title: "क्लिनिकल ग्रेड", desc: "चिकित्सा निरीक्षण के साथ FDA-समकक्ष मानक" },
          { title: "95%+ सटीकता", desc: "विशेषज्ञ रेडियोलॉजिस्ट व्याख्याओं के विरुद्ध मान्य" }
        ]
      }
    }
  };

  const currentContent = content[language];

  return (
    <Container>
      <div className="space-y-xxl">
        {/* Medical Disclaimer Alert */}
        {alertVisible && (
          <Alert 
            type="warning" 
            onClose={() => setAlertVisible(false)}
            className="mt-l"
          >
            {currentContent.alert}
          </Alert>
        )}

        {/* Hero Section */}
        <div className="text-center py-xxl">
          <Card className="relative bg-gradient-to-br from-primary-light/30 to-primary-light/60 border-primary-light/50">
            <div className="flex justify-center mb-xl">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-primary to-primary-dark rounded-card blur-lg opacity-30"></div>
                <div className="relative bg-gradient-to-br from-primary to-primary-dark rounded-card p-l">
                  <svg className="w-16 h-16 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                  </svg>
                </div>
              </div>
            </div>
            
            <Heading1 className="mb-l leading-tight">
              {currentContent.heroTitle}
            </Heading1>
            
            <Heading2 className="mb-l text-primary">
              {currentContent.heroSubtitle}
            </Heading2>
            
            <BodyText className="mb-xl max-w-3xl mx-auto">
              {currentContent.heroDescription}
            </BodyText>
            
            <div className="flex flex-wrap justify-center gap-m text-body-small">
              <span className="bg-secondary-success bg-opacity-20 text-secondary-success px-m py-s rounded-card font-medium">
                ✓ {currentContent.features.fdaGrade}
              </span>
              <span className="bg-primary bg-opacity-20 text-primary px-m py-s rounded-card font-medium">
                ✓ {currentContent.features.realtime}
              </span>
              <span className="bg-primary-dark bg-opacity-20 text-primary-dark px-m py-s rounded-card font-medium">
                ✓ {currentContent.features.hipaa}
              </span>
              <span className="bg-secondary-warning bg-opacity-20 text-secondary-warning px-m py-s rounded-card font-medium">
                ✓ {currentContent.features.accuracy}
              </span>
            </div>
          </Card>
        </div>

        {/* Main Features */}
        <div className="grid md:grid-cols-2 gap-xl">
          {/* Medical Imaging Card */}
          <Card 
            className="group relative overflow-hidden hover:shadow-xl transition-all duration-300 cursor-pointer hover:-translate-y-1 border-2 hover:border-primary-light"
            onClick={() => onNavigate('imageUpload')}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-primary-light/20 group-hover:from-primary/10 group-hover:to-primary-light/30 transition-all duration-300"></div>
            
            <div className="relative p-xl">
              <div className="flex items-center justify-between mb-l">
                <div className="bg-gradient-to-br from-primary to-primary-dark rounded-card p-m group-hover:scale-110 transition-transform duration-300">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                  </svg>
                </div>
                <div className="text-right">
                  <div className="text-h3 font-bold text-neutral-dark-gray group-hover:text-primary transition-colors">{currentContent.imageAnalysis.subtitle}</div>
                </div>
              </div>
              
              <Heading2 className="mb-m group-hover:text-primary transition-colors">
                {currentContent.imageAnalysis.title}
              </Heading2>
              
              <BodyText className="mb-l">
                {currentContent.imageAnalysis.description}
              </BodyText>
              
              <div className="space-y-s mb-l">
                {currentContent.imageAnalysis.features.map((feature, index) => (
                  <div key={index} className="flex items-center text-body-small">
                    <svg className="w-4 h-4 text-secondary-success mr-s" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                    </svg>
                    <span className="text-neutral-dark-gray">{feature}</span>
                  </div>
                ))}
              </div>
              
              <div className="flex items-center text-primary font-semibold group-hover:text-primary-dark">
                <span>{currentContent.imageAnalysis.cta}</span>
                <svg className="w-5 h-5 ml-s group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6"/>
                </svg>
              </div>
            </div>
          </Card>

          {/* AI Consultation Card */}
          <Card 
            className="group relative overflow-hidden hover:shadow-xl transition-all duration-300 cursor-pointer hover:-translate-y-1 border-2 hover:border-secondary-success"
            onClick={() => onNavigate('chatbot')}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-secondary-success/5 to-secondary-success/20 group-hover:from-secondary-success/10 group-hover:to-secondary-success/30 transition-all duration-300"></div>
            
            <div className="relative p-xl">
              <div className="flex items-center justify-between mb-l">
                <div className="bg-gradient-to-br from-secondary-success to-secondary-success rounded-card p-m group-hover:scale-110 transition-transform duration-300">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                  </svg>
                </div>
                <div className="text-right">
                  <div className="text-h3 font-bold text-neutral-dark-gray group-hover:text-secondary-success transition-colors">{currentContent.consultation.subtitle}</div>
                </div>
              </div>
              
              <Heading2 className="mb-m group-hover:text-secondary-success transition-colors">
                {currentContent.consultation.title}
              </Heading2>
              
              <BodyText className="mb-l">
                {currentContent.consultation.description}
              </BodyText>
              
              <div className="space-y-s mb-l">
                {currentContent.consultation.features.map((feature, index) => (
                  <div key={index} className="flex items-center text-body-small">
                    <svg className="w-4 h-4 text-secondary-success mr-s" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                    </svg>
                    <span className="text-neutral-dark-gray">{feature}</span>
                  </div>
                ))}
              </div>
              
              <div className="flex items-center text-secondary-success font-semibold group-hover:text-secondary-success">
                <span>{currentContent.consultation.cta}</span>
                <svg className="w-5 h-5 ml-s group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6"/>
                </svg>
              </div>
            </div>
          </Card>
        </div>

        {/* Technology Showcase */}
        <Card className="bg-gradient-to-r from-neutral-dark-gray to-primary-dark text-white relative overflow-hidden">
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-10 left-10 w-32 h-32 bg-primary/20 rounded-full blur-xl"></div>
            <div className="absolute top-40 right-20 w-24 h-24 bg-secondary-success/20 rounded-full blur-lg"></div>
            <div className="absolute bottom-20 left-32 w-28 h-28 bg-primary-light/20 rounded-full blur-xl"></div>
          </div>
          
          <div className="relative p-xxl">
            <div className="text-center mb-xxl">
              <Heading1 className="mb-m text-white">{currentContent.technology.title}</Heading1>
              <BodyText className="text-xl text-primary-light max-w-3xl mx-auto">
                {currentContent.technology.description}
              </BodyText>
            </div>
            
            <div className="grid md:grid-cols-4 gap-xl">
              {currentContent.technology.features.map((feature, index) => (
                <div key={index} className="text-center">
                  <div className="bg-primary/20 rounded-card w-16 h-16 flex items-center justify-center mx-auto mb-m">
                    <svg className="w-8 h-8 text-primary-light" fill="currentColor" viewBox="0 0 24 24">
                      {index === 0 && <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>}
                      {index === 1 && <path d="M13 3c-4.97 0-9 4.03-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42C8.27 19.99 10.51 21 13 21c4.97 0 9-4.03 9-9s-4.03-9-9-9z"/>}
                      {index === 2 && <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>}
                      {index === 3 && <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>}
                    </svg>
                  </div>
                  <Heading3 className="mb-s text-white">{feature.title}</Heading3>
                  <BodyText variant="secondary" className="text-primary-light">
                    {feature.desc}
                  </BodyText>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>
    </Container>
  );
}
