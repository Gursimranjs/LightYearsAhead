import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Rocket, Telescope, Atom, TrendingUp } from 'lucide-react'

const HomePage = () => {
  const navigate = useNavigate()

  const features = [
    {
      icon: Telescope,
      title: "Exoplanet Classification",
      description: "Upload transit light curve data and get instant classification: CONFIRMED, CANDIDATE, or FALSE POSITIVE with confidence scores"
    },
    {
      icon: Atom,
      title: "Atmospheric Analysis",
      description: "Add spectroscopic data to detect atmospheric gases like H₂O, CH₄, and CO₂ using quantum machine learning"
    }
  ]

  const stats = [
    { label: "F1-Score Accuracy", value: "88.92%" },
    { label: "Gas Detection MAE", value: "5.01%" },
    { label: "Training Time", value: "2 mins" },
    { label: "Targets Analyzed", value: "19,561+" }
  ]

  return (
    <div className="min-h-screen space-bg">
      <div className="relative z-10 min-h-screen">
        {/* Hero Section */}
        <section className="container mx-auto px-4 pt-20 pb-16 md:pt-32 md:pb-24">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center max-w-5xl mx-auto"
          >
            <motion.h1
              className="text-6xl md:text-8xl font-extrabold mb-6 tracking-tight"
              style={{
                textShadow: '0 4px 20px rgba(0,0,0,0.8)',
                color: 'white'
              }}
            >
              LightYears Ahead
            </motion.h1>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.8 }}
              className="text-2xl md:text-3xl text-white/95 font-light mb-12"
              style={{
                textShadow: '0 2px 10px rgba(0,0,0,0.8)'
              }}
            >
              AI-Powered Exoplanet Discovery & Atmospheric Analysis
            </motion.p>

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6, duration: 0.5 }}
              className="flex flex-col sm:flex-row gap-4 justify-center items-center"
            >
              <button
                onClick={() => navigate('/analysis')}
                className="btn-primary text-lg px-10 py-4 flex items-center gap-2 group"
              >
                <Rocket className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                Start Analysis
              </button>

              <a
                href="https://github.com/Gursimranjs/LightYearsAhead"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-secondary text-lg px-10 py-4"
              >
                View on GitHub
              </a>
            </motion.div>
          </motion.div>
        </section>

        {/* Stats Section */}
        <section className="container mx-auto px-4 py-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8, duration: 0.6 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-5xl mx-auto"
          >
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.9 + index * 0.1, duration: 0.5 }}
                className="glass-card p-6 text-center hover:scale-105 transition-transform duration-300"
              >
                <div className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
                  {stat.value}
                </div>
                <div className="text-sm md:text-base text-white/70">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </motion.div>
        </section>

        {/* Features Section */}
        <section className="container mx-auto px-4 py-16">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2, duration: 0.8 }}
            className="max-w-5xl mx-auto"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-center mb-4 text-white">
              What Can You Discover?
            </h2>
            <p className="text-center text-white/70 text-lg mb-12">
              Analyze exoplanet data with NASA-level AI models
            </p>

            <div className="grid md:grid-cols-2 gap-6">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, x: index === 0 ? -30 : 30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 1.4 + index * 0.2, duration: 0.6 }}
                  className="relative group"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-3xl blur-xl group-hover:blur-2xl transition-all duration-300" />
                  <div className="relative glass-card p-8 hover:bg-white/10 transition-all duration-300 border border-indigo-500/30">
                    <feature.icon className="w-12 h-12 text-indigo-400 mb-4" />
                    <h3 className="text-2xl font-bold mb-3 text-indigo-200">
                      {feature.title}
                    </h3>
                    <p className="text-white/80 leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Technology Highlights */}
        <section className="container mx-auto px-4 py-16 pb-24">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.8, duration: 0.8 }}
            className="max-w-4xl mx-auto glass-card p-10 text-center"
          >
            <div className="flex items-center justify-center gap-3 mb-6">
              <TrendingUp className="w-8 h-8 text-green-400" />
              <h3 className="text-3xl font-bold text-white">
                Powered by Advanced AI
              </h3>
            </div>

            <div className="grid md:grid-cols-3 gap-8 text-center">
              <div>
                <div className="text-indigo-400 font-semibold text-lg mb-2">
                  Stage 1
                </div>
                <div className="text-white/90">
                  Transit Classifier
                </div>
                <div className="text-white/60 text-sm mt-1">
                  Stacking Ensemble (RF + XGBoost + LightGBM)
                </div>
              </div>

              <div>
                <div className="text-purple-400 font-semibold text-lg mb-2">
                  Stage 2
                </div>
                <div className="text-white/90">
                  QELM Processor
                </div>
                <div className="text-white/60 text-sm mt-1">
                  12-Qubit Quantum Reservoir Computing
                </div>
              </div>

              <div>
                <div className="text-pink-400 font-semibold text-lg mb-2">
                  Fusion Layer
                </div>
                <div className="text-white/90">
                  Decision Integration
                </div>
                <div className="text-white/60 text-sm mt-1">
                  Transit + Atmospheric Analysis
                </div>
              </div>
            </div>

            <div className="mt-8 pt-6 border-t border-white/10">
              <p className="text-white/60 text-sm">
                NASA Space Apps Challenge 2025 • Best Use of Technology Award Submission
              </p>
            </div>
          </motion.div>
        </section>

        {/* Footer */}
        <footer className="container mx-auto px-4 py-8 text-center text-white/50 text-sm">
          <p className="mb-2">LightYears Ahead - NASA Space Apps Challenge 2025</p>
          <p>Powered by Transit Classifier (88.92% F1) + QELM (5.01% MAE)</p>
        </footer>
      </div>
    </div>
  )
}

export default HomePage
