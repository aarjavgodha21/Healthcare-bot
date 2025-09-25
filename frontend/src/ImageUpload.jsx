import React, { useState, useRef } from "react";

export default function ImageUpload({ onNavigate }) {
  const [imageFile, setImageFile] = useState(null);
  const [imageType, setImageType] = useState("auto");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const imageInputRef = useRef();

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setResult(null);
      setError(null);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setImageFile(file);
      setResult(null);
      setError(null);
    }
  };

  const handleImageTypeChange = (e) => {
    setImageType(e.target.value);
  };

  const handleImageSubmit = async (e) => {
    e.preventDefault();
    if (!imageFile) {
      setError("Please select an image file");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      formData.append("image_type", imageType);
      const res = await fetch("http://127.0.0.1:8001/image-diagnosis", {
        method: "POST",
        body: formData,
        headers: { "Accept": "application/json" },
      });
      if (!res.ok) throw new Error("Network error");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError("Failed to analyze image. Please check if the backend server is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header Section */}
      <div className="bg-white/80 backdrop-blur-lg border-b border-blue-200/50 shadow-lg">
        <div className="max-w-6xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button 
                onClick={() => onNavigate('home')}
                className="group w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-blue-200"
              >
                <svg className="w-6 h-6 text-white group-hover:-translate-x-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/>
                </svg>
              </button>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Medical Imaging Analysis
                </h1>
                <p className="text-sm text-slate-600">AI-Powered Radiological Assessment</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <div className="text-sm font-semibold text-slate-700">18 Pathology Classes</div>
                <div className="text-xs text-slate-500">Advanced Visualization</div>
              </div>
              <div className="w-12 h-12 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-2xl flex items-center justify-center">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl border border-blue-200/50 p-8">
            <div className="text-center mb-8">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-800 mb-2">Upload Medical Image</h2>
              <p className="text-slate-600">Upload X-rays, CT scans, or MRI images for AI analysis</p>
            </div>

            <form onSubmit={handleImageSubmit} className="space-y-6">
              {/* Drag & Drop Area */}
              <div 
                className={`relative border-2 border-dashed rounded-3xl p-12 text-center transition-all duration-300 ${
                  dragActive 
                    ? 'border-blue-400 bg-blue-50' 
                    : imageFile 
                      ? 'border-green-400 bg-green-50' 
                      : 'border-slate-300 hover:border-blue-400 hover:bg-blue-50/50'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  accept="image/*,.nii,.nii.gz"
                  onChange={handleImageChange}
                  ref={imageInputRef}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                
                {imageFile ? (
                  <div className="space-y-4">
                    <div className="w-20 h-20 bg-green-100 rounded-2xl flex items-center justify-center mx-auto">
                      <svg className="w-10 h-10 text-green-600" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                      </svg>
                    </div>
                    <div>
                      <p className="text-lg font-semibold text-green-700">{imageFile.name}</p>
                      <p className="text-sm text-green-600">{(imageFile.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => imageInputRef.current.click()}
                      className="text-blue-600 hover:text-blue-700 font-medium"
                    >
                      Choose different file
                    </button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="w-20 h-20 bg-slate-100 rounded-2xl flex items-center justify-center mx-auto">
                      <svg className="w-10 h-10 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                      </svg>
                    </div>
                    <div>
                      <p className="text-lg font-medium text-slate-700">Drop your medical image here</p>
                      <p className="text-sm text-slate-500">or click to browse files</p>
                    </div>
                    <div className="flex flex-wrap justify-center gap-2 text-xs text-slate-400">
                      <span className="bg-slate-100 px-2 py-1 rounded-full">JPEG</span>
                      <span className="bg-slate-100 px-2 py-1 rounded-full">PNG</span>
                      <span className="bg-slate-100 px-2 py-1 rounded-full">DICOM</span>
                      <span className="bg-slate-100 px-2 py-1 rounded-full">NII</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Image Type Selection */}
              <div className="space-y-3">
                <label className="block text-sm font-semibold text-slate-700">Imaging Modality</label>
                <select 
                  value={imageType} 
                  onChange={handleImageTypeChange} 
                  className="w-full px-4 py-3 rounded-2xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white shadow-sm"
                >
                  <option value="auto">🔍 Auto-detect</option>
                  <option value="xray">📷 X-ray</option>
                  <option value="mri">🧠 MRI</option>
                </select>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                className="w-full py-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-semibold rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105"
                disabled={loading || !imageFile}
              >
                {loading ? (
                  <div className="flex items-center justify-center space-x-3">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Analyzing Image...</span>
                  </div>
                ) : (
                  <div className="flex items-center justify-center space-x-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                    </svg>
                    <span>Analyze Medical Image</span>
                  </div>
                )}
              </button>
            </form>

            {/* Error Display */}
            {error && (
              <div className="mt-6 bg-red-50 border border-red-200 rounded-2xl p-4 flex items-center space-x-3">
                <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                  <svg className="w-4 h-4 text-red-600" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                  </svg>
                </div>
                <div>
                  <div className="text-red-800 font-medium text-sm">Analysis Failed</div>
                  <div className="text-red-600 text-sm">{error}</div>
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl border border-blue-200/50 p-8">
            {result ? (
              <div className="space-y-8">
                {/* Header */}
                <div className="text-center pb-6 border-b border-slate-200">
                  <div className="w-16 h-16 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                    </svg>
                  </div>
                  <h3 className="text-2xl font-bold text-slate-800 mb-2">Analysis Complete</h3>
                  <p className="text-slate-600">AI-powered medical image analysis results</p>
                </div>

                {/* Primary Diagnosis */}
                <div className="bg-gradient-to-r from-emerald-50 to-teal-50 rounded-2xl p-6 border border-emerald-200">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-bold text-emerald-800">Primary Diagnosis</h4>
                    <span className="text-xs font-medium text-emerald-600 bg-emerald-100 px-3 py-1 rounded-full">
                      {result.image_type?.toUpperCase() || 'ANALYSIS'}
                    </span>
                  </div>
                  <div className="space-y-3">
                    <div>
                      <p className="text-2xl font-bold text-emerald-900">{result.prediction}</p>
                      <div className="mt-2 bg-emerald-200 rounded-full h-3 overflow-hidden">
                        <div 
                          className="bg-gradient-to-r from-emerald-500 to-teal-600 h-full rounded-full transition-all duration-1000"
                          style={{ width: `${(result.risk_score * 100)}%` }}
                        ></div>
                      </div>
                      <p className="text-sm text-emerald-700 mt-1 font-medium">
                        Confidence: {(result.risk_score * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>

                {/* Probability Distribution */}
                {result.class_probabilities && (
                  <div className="space-y-4">
                    <h4 className="text-lg font-bold text-slate-800">Differential Diagnosis</h4>
                    <div className="space-y-3">
                      {Object.entries(result.class_probabilities).slice(0, 5).map(([diagnosis, prob]) => (
                        <div key={diagnosis} className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                          <span className="text-sm font-medium text-slate-700">{diagnosis}</span>
                          <div className="flex items-center space-x-3">
                            <div className="w-20 bg-slate-200 rounded-full h-2">
                              <div 
                                className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full transition-all duration-700"
                                style={{ width: `${Math.max((prob * 100), 2)}%` }}
                              ></div>
                            </div>
                            <span className="text-sm font-semibold text-slate-600 w-12 text-right">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Visualizations */}
                {result.visuals && (
                  <div className="space-y-6">
                    <h4 className="text-lg font-bold text-slate-800">Advanced Visualizations</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {Object.entries(result.visuals).slice(0, 4).map(([key, b64], i) => {
                        const vizInfo = result.visualization_info && result.visualization_info[key];
                        return (
                          <div key={key} className="bg-slate-50 rounded-2xl p-4 border border-slate-200">
                            <div className="text-center mb-4">
                              <h5 className="text-sm font-bold text-slate-800 mb-1">
                                {vizInfo ? vizInfo.title : key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
                              </h5>
                              {vizInfo && (
                                <p className="text-xs text-slate-600 leading-relaxed">
                                  {vizInfo.description}
                                </p>
                              )}
                            </div>
                            <div className="flex justify-center">
                              <img
                                src={`data:image/png;base64,${b64}`}
                                alt={vizInfo ? vizInfo.title : key}
                                className="rounded-xl shadow-sm max-w-full h-auto border border-slate-300"
                                style={{ maxHeight: '200px', width: 'auto' }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Clinical Recommendations */}
                {result.suggestions && (
                  <div className="bg-blue-50 rounded-2xl p-6 border border-blue-200">
                    <h4 className="text-lg font-bold text-blue-800 mb-4 flex items-center">
                      <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                      </svg>
                      Clinical Recommendations
                    </h4>
                    <ul className="space-y-2">
                      {result.suggestions.map((tip, i) => (
                        <li key={i} className="flex items-start space-x-3">
                          <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                          <span className="text-sm text-blue-700 leading-relaxed">{tip}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Warnings */}
                {result.warnings && (
                  <div className="bg-amber-50 rounded-2xl p-6 border border-amber-200">
                    <h4 className="text-lg font-bold text-amber-800 mb-4 flex items-center">
                      <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
                      </svg>
                      Important Notices
                    </h4>
                    <ul className="space-y-2">
                      {Object.entries(result.warnings).map(([k, v]) => (
                        <li key={k} className="flex items-start space-x-3">
                          <div className="w-1.5 h-1.5 bg-amber-500 rounded-full mt-2 flex-shrink-0"></div>
                          <span className="text-sm text-amber-700 leading-relaxed">{v}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center py-16">
                <div className="w-24 h-24 bg-slate-100 rounded-3xl flex items-center justify-center mb-6">
                  <svg className="w-12 h-12 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-slate-700 mb-4">Ready for Analysis</h3>
                <p className="text-slate-500 max-w-sm leading-relaxed">
                  Upload a medical image to begin AI-powered analysis. Results will appear here with detailed visualizations and clinical insights.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
