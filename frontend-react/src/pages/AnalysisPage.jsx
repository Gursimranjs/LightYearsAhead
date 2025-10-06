import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import {
  Upload, FileText, CheckCircle, XCircle, ArrowLeft,
  AlertCircle, Sparkles, Database, Atom, ChevronRight
} from 'lucide-react'
import axios from 'axios'

const AnalysisPage = () => {
  const navigate = useNavigate()
  const [transitFile, setTransitFile] = useState(null)
  const [spectrumFile, setSpectrumFile] = useState(null)
  const [transitPreview, setTransitPreview] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState(null)
  const [currentStep, setCurrentStep] = useState(1)

  // Dropzone for transit data
  const transitDropzone = useDropzone({
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0]
        setTransitFile(file)
        setError(null)

        // Read CSV for preview
        const reader = new FileReader()
        reader.onload = (e) => {
          const text = e.target.result
          const lines = text.split('\n').slice(0, 4)
          const headers = lines[0].split(',')
          const rows = lines.slice(1).map(line => line.split(','))
          setTransitPreview({ headers, rows, totalRows: text.split('\n').length - 1 })
        }
        reader.readAsText(file)
      }
    }
  })

  // Dropzone for spectrum data
  const spectrumDropzone = useDropzone({
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setSpectrumFile(acceptedFiles[0])
      }
    }
  })

  const handleAnalyze = async () => {
    if (!transitFile) {
      setError('Please upload transit data file')
      return
    }

    setIsAnalyzing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', transitFile)

      const response = await axios.post('/api/upload-csv', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000
      })

      if (response.data.success) {
        navigate('/results', { state: { results: response.data } })
      } else {
        setError(response.data.error || 'Analysis failed')
      }
    } catch (err) {
      console.error('Analysis error:', err)
      setError(err.response?.data?.error || 'Failed to connect to backend. Make sure the API is running on port 5001.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const steps = [
    { number: 1, title: 'Upload Data', icon: Upload },
    { number: 2, title: 'Analyze', icon: Sparkles },
    { number: 3, title: 'Results', icon: Database }
  ]

  return (
    <div className="min-h-screen space-bg">
      <div className="relative z-10 min-h-screen">
        {/* Header */}
        <div className="container mx-auto px-4 pt-8">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 text-white/70 hover:text-white transition-colors mb-6"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Home
          </button>
        </div>

        {/* Progress Steps */}
        <div className="container mx-auto px-4 py-8">
          <div className="flex justify-center items-center gap-4 max-w-2xl mx-auto">
            {steps.map((step, index) => (
              <div key={step.number} className="flex items-center gap-4">
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center gap-3"
                >
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center font-bold transition-all duration-300 ${
                      currentStep >= step.number
                        ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg'
                        : 'bg-white/10 text-white/40'
                    }`}
                  >
                    <step.icon className="w-6 h-6" />
                  </div>
                  <div>
                    <div className={`text-sm font-semibold ${
                      currentStep >= step.number ? 'text-white' : 'text-white/40'
                    }`}>
                      {step.title}
                    </div>
                  </div>
                </motion.div>
                {index < steps.length - 1 && (
                  <ChevronRight className={`w-5 h-5 ${
                    currentStep > step.number ? 'text-purple-400' : 'text-white/20'
                  }`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-4 pb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-5xl mx-auto"
          >
            <h1 className="text-4xl md:text-5xl font-bold text-center mb-4 text-white">
              Upload Exoplanet Data
            </h1>
            <p className="text-center text-white/70 text-lg mb-12">
              Upload your transit data to analyze potential exoplanets
            </p>

            {/* Error Message */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="glass-card bg-red-500/20 border-red-500/50 p-4 mb-6 flex items-center gap-3"
                >
                  <XCircle className="w-5 h-5 text-red-400" />
                  <p className="text-red-200">{error}</p>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="grid md:grid-cols-2 gap-6 mb-8">
              {/* Transit Data Upload */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                className="glass-card-solid p-6"
              >
                <div className="flex items-center gap-3 mb-4">
                  <Database className="w-6 h-6 text-blue-400" />
                  <h3 className="text-xl font-bold text-white">
                    Transit Data (Required)
                  </h3>
                </div>

                <div className="glass-card bg-blue-500/10 border-blue-500/30 p-4 mb-4">
                  <p className="text-sm text-white/80 mb-2">
                    <strong className="text-blue-300">Purpose:</strong> Detects planets by measuring the dimming of starlight
                  </p>
                  <p className="text-sm text-white/70">
                    <strong className="text-blue-300">Required:</strong> CSV with period, depth, duration, snr, prad, teq, etc.
                  </p>
                </div>

                <div
                  {...transitDropzone.getRootProps()}
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 ${
                    transitDropzone.isDragActive
                      ? 'border-blue-400 bg-blue-500/10'
                      : transitFile
                      ? 'border-green-400 bg-green-500/10'
                      : 'border-white/20 bg-white/5 hover:border-blue-400 hover:bg-blue-500/5'
                  }`}
                >
                  <input {...transitDropzone.getInputProps()} />
                  <div className="flex flex-col items-center gap-3">
                    {transitFile ? (
                      <>
                        <CheckCircle className="w-12 h-12 text-green-400" />
                        <div>
                          <p className="text-white font-semibold">{transitFile.name}</p>
                          <p className="text-white/60 text-sm">
                            {(transitFile.size / 1024).toFixed(2)} KB
                          </p>
                        </div>
                      </>
                    ) : (
                      <>
                        <Upload className="w-12 h-12 text-white/40" />
                        <div>
                          <p className="text-white/80">Drop CSV file here or click to browse</p>
                          <p className="text-white/50 text-sm mt-1">Maximum file size: 10MB</p>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Preview */}
                {transitPreview && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-4 glass-card p-4"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-semibold text-white flex items-center gap-2">
                        <FileText className="w-4 h-4" />
                        Data Preview
                      </h4>
                      <span className="text-xs text-white/60">
                        {transitPreview.totalRows} rows × {transitPreview.headers.length} columns
                      </span>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-white/10">
                            {transitPreview.headers.slice(0, 4).map((header, i) => (
                              <th key={i} className="text-left py-2 px-2 text-white/70 font-semibold">
                                {header}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {transitPreview.rows.map((row, i) => (
                            <tr key={i} className="border-b border-white/5">
                              {row.slice(0, 4).map((cell, j) => (
                                <td key={j} className="py-2 px-2 text-white/80">
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </motion.div>
                )}
              </motion.div>

              {/* Spectrum Data Upload */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
                className="glass-card-solid p-6"
              >
                <div className="flex items-center gap-3 mb-4">
                  <Atom className="w-6 h-6 text-purple-400" />
                  <h3 className="text-xl font-bold text-white">
                    Spectrum Data (Optional)
                  </h3>
                </div>

                <div className="glass-card bg-purple-500/10 border-purple-500/30 p-4 mb-4">
                  <p className="text-sm text-white/80 mb-2">
                    <strong className="text-purple-300">Purpose:</strong> Reveals atmospheric composition
                  </p>
                  <p className="text-sm text-white/70 mb-2">
                    <strong className="text-purple-300">Detects:</strong> H₂O (water), CH₄ (methane), CO₂
                  </p>
                  <p className="text-sm text-white/60">
                    <strong className="text-purple-300">Impact:</strong> Increases confidence + provides gas abundances
                  </p>
                </div>

                <div
                  {...spectrumDropzone.getRootProps()}
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 ${
                    spectrumDropzone.isDragActive
                      ? 'border-purple-400 bg-purple-500/10'
                      : spectrumFile
                      ? 'border-green-400 bg-green-500/10'
                      : 'border-white/20 bg-white/5 hover:border-purple-400 hover:bg-purple-500/5'
                  }`}
                >
                  <input {...spectrumDropzone.getInputProps()} />
                  <div className="flex flex-col items-center gap-3">
                    {spectrumFile ? (
                      <>
                        <CheckCircle className="w-12 h-12 text-green-400" />
                        <div>
                          <p className="text-white font-semibold">{spectrumFile.name}</p>
                          <p className="text-white/60 text-sm">
                            {(spectrumFile.size / 1024).toFixed(2)} KB
                          </p>
                        </div>
                      </>
                    ) : (
                      <>
                        <Upload className="w-12 h-12 text-white/40" />
                        <div>
                          <p className="text-white/80">Drop CSV file here (optional)</p>
                          <p className="text-white/50 text-sm mt-1">Wavelengths: 0.6-2.8 μm</p>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {spectrumFile && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-4 glass-card bg-green-500/10 border-green-500/30 p-3 flex items-center gap-2"
                  >
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <p className="text-sm text-green-200">
                      Atmospheric analysis will be included!
                    </p>
                  </motion.div>
                )}
              </motion.div>
            </div>

            {/* Analyze Button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="flex justify-center"
            >
              <button
                onClick={handleAnalyze}
                disabled={!transitFile || isAnalyzing}
                className={`btn-primary text-lg px-12 py-4 flex items-center gap-3 ${
                  (!transitFile || isAnalyzing) ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full spinner" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Analyze Data
                  </>
                )}
              </button>
            </motion.div>

            {/* Info */}
            {isAnalyzing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-6 glass-card bg-blue-500/10 border-blue-500/30 p-4 max-w-2xl mx-auto text-center"
              >
                <div className="flex items-center justify-center gap-2 text-blue-200">
                  <AlertCircle className="w-5 h-5" />
                  <p className="text-sm">
                    Processing your data through our AI models. This may take a moment...
                  </p>
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  )
}

export default AnalysisPage
