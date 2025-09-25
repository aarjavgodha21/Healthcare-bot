import React, { useState } from 'react';

// Design System Components for Diagnogenie

// Button Components
export const Button = ({ 
  variant = 'primary', 
  children, 
  disabled = false, 
  onClick, 
  className = '',
  ...props 
}) => {
  const baseClasses = 'px-l py-m rounded-button font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2';
  
  const variants = {
    primary: 'bg-primary text-white hover:brightness-105 disabled:bg-neutral-light-gray disabled:text-neutral-medium-gray disabled:cursor-not-allowed',
    secondary: 'bg-transparent text-primary border-2 border-primary hover:bg-primary-light disabled:bg-neutral-light-gray disabled:text-neutral-medium-gray disabled:border-neutral-light-gray disabled:cursor-not-allowed',
    danger: 'bg-secondary-error text-white hover:brightness-105 disabled:bg-neutral-light-gray disabled:text-neutral-medium-gray disabled:cursor-not-allowed'
  };

  return (
    <button
      className={`${baseClasses} ${variants[variant]} ${className}`}
      disabled={disabled}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
};

// Typography Components
export const Heading1 = ({ children, className = '', ...props }) => (
  <h1 className={`text-h1 font-bold text-neutral-dark-gray ${className}`} {...props}>
    {children}
  </h1>
);

export const Heading2 = ({ children, className = '', ...props }) => (
  <h2 className={`text-h2 font-semibold text-neutral-dark-gray ${className}`} {...props}>
    {children}
  </h2>
);

export const Heading3 = ({ children, className = '', ...props }) => (
  <h3 className={`text-h3 font-medium text-neutral-dark-gray ${className}`} {...props}>
    {children}
  </h3>
);

export const BodyText = ({ children, variant = 'primary', className = '', ...props }) => {
  const variants = {
    primary: 'text-body text-neutral-dark-gray',
    secondary: 'text-body-small text-neutral-medium-gray'
  };

  return (
    <p className={`${variants[variant]} ${className}`} {...props}>
      {children}
    </p>
  );
};

export const Caption = ({ children, className = '', ...props }) => (
  <span className={`text-caption font-medium text-neutral-medium-gray ${className}`} {...props}>
    {children}
  </span>
);

// Alert Components
export const Alert = ({ type = 'info', children, className = '', onClose, ...props }) => {
  const types = {
    info: 'bg-alert-info-bg text-alert-info-text',
    warning: 'bg-alert-warning-bg text-alert-warning-text',
    error: 'bg-alert-error-bg text-alert-error-text',
    success: 'bg-alert-success-bg text-alert-success-text'
  };

  return (
    <div className={`rounded-button p-m my-s font-medium ${types[type]} ${className}`} {...props}>
      <div className="flex items-center justify-between">
        <div>{children}</div>
        {onClose && (
          <button 
            onClick={onClose}
            className="ml-m text-current opacity-70 hover:opacity-100"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
};

// Chat Components
export const MedicalReport = ({ sections = [], className = '' }) => {
  if (!sections || sections.length === 0) {
    return <div className="text-neutral-medium-gray">No medical information available.</div>;
  }

  return (
    <div className={`medical-report space-y-6 ${className}`}>
      {sections.map((section, index) => {
        if (section.type === 'main-header') {
          return (
            <div key={index} className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-primary to-primary-dark rounded-full mb-4">
                <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                </svg>
              </div>
              <h1 className="text-2xl font-bold text-neutral-dark-gray">{section.text}</h1>
            </div>
          );
        }

        if (section.type === 'section') {
          return (
            <div key={index} className="bg-white rounded-xl border border-neutral-light-gray p-6 shadow-sm">
              <h2 className="text-lg font-semibold text-neutral-dark-gray mb-4 flex items-center">
                {section.title}
              </h2>
              
              <div className="space-y-3">
                {section.items?.map((item, itemIndex) => {
                  if (item.type === 'highlight') {
                    return (
                      <div key={itemIndex} className="bg-gradient-to-r from-primary/10 to-primary-dark/10 rounded-lg p-4 border-l-4 border-primary">
                        <p className="text-neutral-dark-gray font-medium">{item.text}</p>
                      </div>
                    );
                  }

                  if (item.type === 'bullet') {
                    return (
                      <div key={itemIndex} className="flex items-start space-x-3">
                        <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                        <p className="text-neutral-dark-gray leading-relaxed">{item.text}</p>
                      </div>
                    );
                  }

                  if (item.type === 'text') {
                    return (
                      <p key={itemIndex} className="text-neutral-dark-gray leading-relaxed">
                        {item.text}
                      </p>
                    );
                  }

                  return null;
                })}
              </div>
            </div>
          );
        }

        return null;
      })}
    </div>
  );
};

export const ChatBubble = ({ 
  role = 'assistant', 
  content = '', 
  formattedSections = null,
  language = 'en',
  timestamp = 'Just now',
  onSpeak,
  isSpeaking = false,
  onStopSpeak,
  showLanguageToggle = false,
  className = '', 
  ...props 
}) => {
  const isUser = role === 'user';
  const [selectedTTSLanguage, setSelectedTTSLanguage] = useState(language === 'hi' ? 'hi-IN' : 'en-US');

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} ${className}`} {...props}>
      <div className={`max-w-4xl flex ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-4`}>
        {/* Avatar */}
        <div className={`w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0 shadow-lg ${
          isUser
            ? 'bg-gradient-to-br from-primary to-primary-dark ml-4'
            : 'bg-gradient-to-br from-secondary-success to-secondary-success/80 mr-4'
        }`}>
          {isUser ? (
            <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
            </svg>
          ) : (
            <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
            </svg>
          )}
        </div>
        
        {/* Message Content */}
        <div className={`relative rounded-2xl shadow-lg max-w-3xl ${
          isUser
            ? 'bg-gradient-to-br from-primary to-primary-dark text-white px-6 py-4'
            : 'bg-white border border-neutral-light-gray text-neutral-dark-gray px-5 py-4'
        } ${language === 'hi' ? 'font-hindi' : 'font-primary'}`}>

          {isUser ? (
            <div className="whitespace-pre-wrap text-sm leading-relaxed">
              {content}
            </div>
          ) : (
            formattedSections && formattedSections.length > 0 ? (
              <div className="space-y-2">
                {formattedSections.map((sec, idx) => (
                  <div
                    key={idx}
                    className={`pl-4 border-l-2 ${
                      (sec.title && (sec.title.includes('Red Flags') || sec.title.includes('खतरे'))) ? 'border-secondary-error'
                      : (sec.title && (sec.title.includes('Disclaimer') || sec.title.includes('अस्वीकरण'))) ? 'border-neutral-light-gray'
                      : 'border-primary/60'
                    }`}
                  >
                    {sec.title && (
                      <div className="text-[0.95rem] md:text-base font-semibold text-neutral-dark-gray mb-1 flex items-center">
                        <span>{sec.title}</span>
                      </div>
                    )}
                    <div className="space-y-[6px]">
                      {sec.items?.map((it, j) => (
                        it.type === 'bullet' ? (
                          <div key={j} className="flex items-start text-sm leading-relaxed">
                            <span className="mt-1 mr-2 inline-block w-1.5 h-1.5 rounded-full bg-primary"></span>
                            <span>{it.text}</span>
                          </div>
                        ) : it.type === 'highlight' ? (
                          <div key={j} className="text-sm leading-relaxed bg-primary/5 border border-primary/20 rounded-md px-3 py-2">
                            {it.text}
                          </div>
                        ) : (
                          <div key={j} className="text-sm leading-relaxed">
                            {it.text}
                          </div>
                        )
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {content}
              </div>
            )
          )}
          
          {/* Assistant Controls */}
          {!isUser && showLanguageToggle && (
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-neutral-light-gray">
              <div className="flex items-center space-x-2 text-xs text-neutral-medium-gray">
                <div className="w-1.5 h-1.5 bg-secondary-success rounded-full"></div>
                <span>AI Medical Assistant</span>
              </div>
              <div className="flex items-center space-x-2">
                {/* Language Selection for TTS */}
                <select
                  value={selectedTTSLanguage}
                  onChange={(e) => setSelectedTTSLanguage(e.target.value)}
                  className="text-xs bg-neutral-off-white border border-neutral-light-gray rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary"
                >
                  <option value="en-US">🇺🇸 English</option>
                  <option value="hi-IN">🇮🇳 Hindi</option>
                </select>
                
                {/* Text to Speech Controls */}
                <div className="flex items-center space-x-1">
                  <button
                    type="button"
                    onClick={() => onSpeak && onSpeak(selectedTTSLanguage)}
                    className="p-1.5 rounded-lg bg-secondary-success/10 text-secondary-success hover:bg-secondary-success/20 transition-all text-xs"
                    disabled={isSpeaking}
                    title={`Listen in ${selectedTTSLanguage === 'hi-IN' ? 'Hindi' : 'English'}`}
                  >
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
                    </svg>
                  </button>
                  {isSpeaking && onStopSpeak && (
                    <button
                      type="button"
                      onClick={onStopSpeak}
                      className="p-1.5 rounded-lg bg-secondary-error/10 text-secondary-error hover:bg-secondary-error/20 transition-all text-xs"
                      title="Stop speaking"
                    >
                      <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M6 6h12v12H6z"/>
                      </svg>
                    </button>
                  )}
                </div>
                <div className="text-xs text-neutral-medium-gray">
                  {timestamp}
                </div>
              </div>
            </div>
          )}
          
          {/* User Query Label */}
          {isUser && (
            <div className="text-xs text-primary-light mt-2 text-right">
              {language === 'en' ? 'Patient Query' : 'रोगी प्रश्न'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Image Upload Component
export const ImageUpload = ({ 
  onFileSelect, 
  accept = "image/*", 
  status = 'default', 
  className = '',
  ...props 
}) => {
  const [dragOver, setDragOver] = useState(false);
  
  const statusClasses = {
    default: 'border-neutral-light-gray border-dashed hover:border-primary',
    success: 'border-secondary-success bg-secondary-success bg-opacity-10',
    error: 'border-secondary-error bg-secondary-error bg-opacity-10'
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && onFileSelect) {
      onFileSelect(files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    if (file && onFileSelect) {
      onFileSelect(file);
    }
  };

  return (
    <div 
      className={`
        border-2 rounded-card p-xl text-center cursor-pointer transition-colors
        ${statusClasses[status]} 
        ${dragOver ? 'border-primary bg-primary-light' : ''}
        ${className}
      `}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      {...props}
    >
      <input
        type="file"
        accept={accept}
        onChange={handleFileInput}
        className="hidden"
        id="file-upload"
      />
      <label htmlFor="file-upload" className="cursor-pointer">
        <div className="text-primary text-4xl mb-m">📁</div>
        <BodyText>Click or drag to upload an image</BodyText>
        {status === 'success' && (
          <div className="text-secondary-success text-2xl mt-s">✓</div>
        )}
        {status === 'error' && (
          <div className="text-secondary-error text-2xl mt-s">⚠</div>
        )}
      </label>
    </div>
  );
};

// Language Toggle Component
export const LanguageToggle = ({ currentLanguage = 'en', onLanguageChange, className = '' }) => {
  const languages = [
    { code: 'en', label: 'English', flag: '🇺🇸' },
    { code: 'hi', label: 'हिंदी', flag: '🇮🇳' }
  ];

  return (
    <div className={`fixed top-m right-m bg-white border border-neutral-light-gray rounded-button px-m py-s shadow-sm z-50 ${className}`}>
      <select 
        value={currentLanguage}
        onChange={(e) => onLanguageChange && onLanguageChange(e.target.value)}
        className="bg-transparent border-none outline-none cursor-pointer text-body-small"
      >
        {languages.map(lang => (
          <option key={lang.code} value={lang.code}>
            {lang.flag} {lang.label}
          </option>
        ))}
      </select>
    </div>
  );
};

// Card Component
export const Card = ({ children, className = '', ...props }) => (
  <div 
    className={`bg-white rounded-card p-l shadow-md border border-neutral-light-gray ${className}`}
    {...props}
  >
    {children}
  </div>
);

// Container Component
export const Container = ({ children, className = '', ...props }) => (
  <div className={`container mx-auto px-m ${className}`} {...props}>
    {children}
  </div>
);

// Modal Component
export const Modal = ({ isOpen, onClose, children, title, className = '' }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className={`bg-white rounded-modal p-xl max-w-lg w-full mx-m animate-slide-in-bottom ${className}`}>
        {title && (
          <div className="flex items-center justify-between mb-l">
            <Heading3>{title}</Heading3>
            <button 
              onClick={onClose}
              className="text-neutral-medium-gray hover:text-neutral-dark-gray text-2xl"
            >
              ×
            </button>
          </div>
        )}
        {children}
      </div>
    </div>
  );
};

// Voice Button Component
export const VoiceButton = ({ isListening = false, onClick, className = '' }) => (
  <button
    onClick={onClick}
    className={`
      w-12 h-12 rounded-full bg-primary text-white flex items-center justify-center
      hover:brightness-105 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2
      ${isListening ? 'animate-pulse' : ''}
      ${className}
    `}
  >
    🎤
  </button>
);

// Spacer Component
export const Spacer = ({ size = 'm' }) => {
  const sizes = {
    xs: 'h-xs',
    s: 'h-s',
    m: 'h-m',
    l: 'h-l',
    xl: 'h-xl',
    xxl: 'h-xxl'
  };

  return <div className={sizes[size]} />;
};

// Loading Spinner
export const LoadingSpinner = ({ size = 'md', color = 'primary', className = '' }) => {
  const sizes = {
    xs: 'w-3 h-3',
    small: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  };

  const colors = {
    primary: 'border-primary',
    white: 'border-white border-opacity-30 border-t-white',
    success: 'border-secondary-success',
    error: 'border-secondary-error'
  };

  return (
    <div className={`${sizes[size]} ${className}`}>
      <div className={`animate-spin rounded-full border-2 border-neutral-light-gray ${colors[color]} border-t-transparent`}></div>
    </div>
  );
};

// Input Component
export const Input = ({ 
  label, 
  error, 
  hint,
  className = '',
  ...props 
}) => (
  <div className="mb-m">
    {label && (
      <label className="block text-body font-medium text-neutral-dark-gray mb-s">
        {label}
      </label>
    )}
    <input
      className={`
        w-full px-m py-s border border-neutral-light-gray rounded-button
        focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent
        disabled:bg-neutral-off-white disabled:text-neutral-medium-gray
        ${error ? 'border-secondary-error' : ''}
        ${className}
      `}
      {...props}
    />
    {hint && !error && (
      <Caption className="mt-xs">{hint}</Caption>
    )}
    {error && (
      <Caption className="mt-xs text-secondary-error">{error}</Caption>
    )}
  </div>
);

export default { 
  Button, 
  Heading1, 
  Heading2, 
  Heading3, 
  BodyText, 
  Caption, 
  Alert, 
  ChatBubble, 
  MedicalReport,
  ImageUpload, 
  LanguageToggle, 
  Card, 
  Container, 
  Modal, 
  VoiceButton, 
  Spacer, 
  LoadingSpinner, 
  Input 
};