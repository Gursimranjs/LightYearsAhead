import { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ArrowLeft, Download, CheckCircle, XCircle, AlertCircle,
  TrendingUp, Database, Sparkles, FileDown, Package
} from 'lucide-react'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import axios from 'axios'

const ResultsPage = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const [results, setResults] = useState(null)
  const [selectedTarget, setSelectedTarget] = useState(null)
  const [downloading, setDownloading] = useState(false)

  useEffect(() => {
    if (location.state?.results) {
      setResults(location.state.results)
      if (location.state.results.results?.length > 0) {
        setSelectedTarget(0)
      }
    } else {
      navigate('/analysis')
    }
  }, [location, navigate])

  if (!results) return null

  const classificationData = [
    {
      name: 'Confirmed',
      value: results.results?.filter(r => r.classification === 'CONFIRMED').length || 0,
      color: '#10b981'
    },
    {
      name: 'Candidate',
      value: results.results?.filter(r => r.classification === 'CANDIDATE').length || 0,
      color: '#3b82f6'
    },
    {
      name: 'False Positive',
      value: results.results?.filter(r => r.classification === 'FALSE POSITIVE').length || 0,
      color: '#ef4444'
    }
  ]

  const confidenceData = results.results?.map((r, i) => ({
    name: r.target_name?.substring(0, 15) || `Target ${i + 1}`,
    confidence: (r.confidence * 100).toFixed(1),
    quality: r.quality_score
  })) || []

  const handleDownloadReport = async (index) => {
    setDownloading(true)
    try {
      const response = await axios.get(
        `/api/download-report/${results.session_id}/${index}`,
        { responseType: 'blob' }
      )

      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `${results.results[index].target_name}_report.pdf`)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (err) {
      console.error('Download error:', err)
      alert('Failed to download report')
    } finally {
      setDownloading(false)
    }
  }

  const handleDownloadAll = async () => {
    setDownloading(true)
    try {
      const response = await axios.get(
        `/api/download-all-reports/${results.session_id}`,
        { responseType: 'blob' }
      )

      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `exoplanet_reports_${results.session_id}.zip`)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (err) {
      console.error('Download error:', err)
      alert('Failed to download ZIP file')
    } finally {
      setDownloading(false)
    }
  }

  const getClassificationBadge = (classification) => {
    const badges = {
      'CONFIRMED': 'bg-green-500 text-white',
      'CANDIDATE': 'bg-blue-500 text-white',
      'FALSE POSITIVE': 'bg-red-500 text-white'
    }
    return badges[classification] || 'bg-gray-500 text-white'
  }

  const selectedResult = selectedTarget !== null ? results.results[selectedTarget] : null

  return (
    <div className="min-h-screen space-bg">
      <div className="relative z-10 min-h-screen">
        {/* Header */}
        <div className="container mx-auto px-4 pt-8">
          <button
            onClick={() => navigate('/analysis')}
            className="flex items-center gap-2 text-white/70 hover:text-white transition-colors mb-6"
          >
            <ArrowLeft className="w-5 h-5" />
            Analyze New Data
          </button>
        </div>

        {/* Title */}
        <div className="container mx-auto px-4 py-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <div className="flex items-center justify-center gap-3 mb-4">
              <CheckCircle className="w-12 h-12 text-green-400" />
              <h1 className="text-4xl md:text-5xl font-bold text-white">
                Analysis Complete!
              </h1>
            </div>
            <p className="text-white/70 text-lg">
              Your exoplanet data has been analyzed using our AI models
            </p>
          </motion.div>

          {/* Summary Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto mb-8"
          >
            <div className="glass-card p-6 text-center">
              <Database className="w-8 h-8 text-blue-400 mx-auto mb-2" />
              <div className="text-3xl font-bold text-white mb-1">
                {results.total_targets}
              </div>
              <div className="text-sm text-white/60">Total Targets</div>
            </div>

            <div className="glass-card p-6 text-center">
              <CheckCircle className="w-8 h-8 text-green-400 mx-auto mb-2" />
              <div className="text-3xl font-bold text-white mb-1">
                {results.successful}
              </div>
              <div className="text-sm text-white/60">Successful</div>
            </div>

            <div className="glass-card p-6 text-center">
              <XCircle className="w-8 h-8 text-red-400 mx-auto mb-2" />
              <div className="text-3xl font-bold text-white mb-1">
                {results.failed}
              </div>
              <div className="text-sm text-white/60">Failed</div>
            </div>

            <div className="glass-card p-6 text-center">
              <TrendingUp className="w-8 h-8 text-purple-400 mx-auto mb-2" />
              <div className="text-3xl font-bold text-white mb-1">
                {results.average_quality.toFixed(0)}%
              </div>
              <div className="text-sm text-white/60">Avg Quality</div>
            </div>
          </motion.div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-4 pb-16">
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            {/* Classification Distribution */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-card-solid p-6"
            >
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-yellow-400" />
                Classification Distribution
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={classificationData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={false}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {classificationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Legend
                    verticalAlign="bottom"
                    height={36}
                    formatter={(value, entry) => (
                      <span style={{ color: 'white' }}>{`${value}: ${entry.payload.value}`}</span>
                    )}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(15, 23, 42, 0.9)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '8px',
                      color: 'white'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Confidence Scores */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-card-solid p-6"
            >
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                Confidence Scores
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={confidenceData.slice(0, 5)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis
                    dataKey="name"
                    stroke="#fff"
                    tick={{ fill: '#fff', fontSize: 12 }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis stroke="#fff" tick={{ fill: '#fff' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(15, 23, 42, 0.9)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '8px',
                      color: 'white'
                    }}
                  />
                  <Bar dataKey="confidence" fill="#6366f1" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </motion.div>
          </div>

          {/* Detailed Results */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="glass-card-solid p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                <Database className="w-6 h-6 text-blue-400" />
                Detailed Results
              </h3>
              <button
                onClick={handleDownloadAll}
                disabled={downloading}
                className="btn-secondary flex items-center gap-2"
              >
                <Package className="w-4 h-4" />
                Download All (ZIP)
              </button>
            </div>

            <div className="grid md:grid-cols-3 gap-4 mb-6">
              {results.results?.map((result, index) => (
                <motion.button
                  key={index}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.5 + index * 0.05 }}
                  onClick={() => setSelectedTarget(index)}
                  className={`glass-card p-4 text-left transition-all duration-300 ${
                    selectedTarget === index
                      ? 'bg-indigo-500/20 border-indigo-500/50'
                      : 'hover:bg-white/10'
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-semibold text-white text-sm">
                      {result.target_name || `Target ${index + 1}`}
                    </h4>
                    {result.success ? (
                      <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                    )}
                  </div>
                  {result.success && (
                    <>
                      <div className={`inline-block px-2 py-1 rounded text-xs font-bold mb-2 ${
                        getClassificationBadge(result.classification)
                      }`}>
                        {result.classification}
                      </div>
                      <div className="text-xs text-white/60">
                        Confidence: {(result.confidence * 100).toFixed(1)}%
                      </div>
                    </>
                  )}
                </motion.button>
              ))}
            </div>

            {/* Selected Target Details */}
            <AnimatePresence mode="wait">
              {selectedResult && (
                <motion.div
                  key={selectedTarget}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="bg-gradient-to-br from-slate-800 to-slate-900 border border-white/20 p-6 rounded-2xl"
                >
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h4 className="text-2xl font-bold text-white mb-2">
                        {selectedResult.target_name || `Target ${selectedTarget + 1}`}
                      </h4>
                      <div className={`inline-block px-3 py-1 rounded-lg text-sm font-bold ${
                        getClassificationBadge(selectedResult.classification)
                      }`}>
                        {selectedResult.classification}
                      </div>
                    </div>
                    <button
                      onClick={() => handleDownloadReport(selectedTarget)}
                      disabled={downloading}
                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg flex items-center gap-2 transition-colors"
                    >
                      <FileDown className="w-4 h-4" />
                      Download PDF
                    </button>
                  </div>

                  {selectedResult.success ? (
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <h5 className="font-semibold text-white mb-3">Classification Details</h5>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-white/70">Confidence:</span>
                            <span className="font-semibold text-white">
                              {(selectedResult.confidence * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="w-full bg-white/10 rounded-full h-2">
                            <div
                              className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${selectedResult.confidence * 100}%` }}
                            />
                          </div>
                        </div>

                        <h5 className="font-semibold text-white mt-6 mb-3">Probabilities</h5>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-white/70">🟢 Confirmed:</span>
                            <span className="font-semibold text-white">
                              {(selectedResult.probabilities.CONFIRMED * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-white/70">🔵 Candidate:</span>
                            <span className="font-semibold text-white">
                              {(selectedResult.probabilities.CANDIDATE * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-white/70">🔴 False Positive:</span>
                            <span className="font-semibold text-white">
                              {(selectedResult.probabilities['FALSE POSITIVE'] * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h5 className="font-semibold text-white mb-3">Data Quality</h5>
                        <div className="flex items-center gap-4 mb-4">
                          <div className="flex-1">
                            <div className="text-3xl font-bold text-white">
                              {selectedResult.quality_score.toFixed(1)}%
                            </div>
                            <div className="text-sm text-white/70">Overall Quality</div>
                          </div>
                          <div className="w-24 h-24">
                            <ResponsiveContainer width="100%" height="100%">
                              <PieChart>
                                <Pie
                                  data={[
                                    { value: selectedResult.quality_score },
                                    { value: 100 - selectedResult.quality_score }
                                  ]}
                                  cx="50%"
                                  cy="50%"
                                  innerRadius={20}
                                  outerRadius={35}
                                  dataKey="value"
                                  startAngle={90}
                                  endAngle={-270}
                                >
                                  <Cell fill="#6366f1" />
                                  <Cell fill="rgba(255,255,255,0.1)" />
                                </Pie>
                              </PieChart>
                            </ResponsiveContainer>
                          </div>
                        </div>

                        {selectedResult.atmospheric_data && (
                          <div className="mt-6">
                            <h5 className="font-semibold text-white mb-3">
                              Atmospheric Composition
                            </h5>
                            <div className="space-y-2">
                              <div className="flex justify-between">
                                <span className="text-white/70">💧 H₂O (Water):</span>
                                <span className="font-semibold text-white">
                                  {(selectedResult.atmospheric_data.h2o * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-white/70">🔥 CH₄ (Methane):</span>
                                <span className="font-semibold text-white">
                                  {(selectedResult.atmospheric_data.ch4 * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-white/70">💨 CO₂:</span>
                                <span className="font-semibold text-white">
                                  {(selectedResult.atmospheric_data.co2 * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                      <AlertCircle className="w-5 h-5 text-red-500" />
                      <div>
                        <p className="font-semibold text-red-900">Analysis Failed</p>
                        <p className="text-sm text-red-700">
                          {selectedResult.error || 'Unknown error occurred'}
                        </p>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

export default ResultsPage
