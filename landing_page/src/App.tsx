import { motion } from 'framer-motion'
import Hero from './components/Hero'
import Features from './components/Features'
import Metrics from './components/Metrics'
import HowItWorks from './components/HowItWorks'
import CallToAction from './components/CallToAction'
import Footer from './components/Footer'
import BackgroundEffects from './components/BackgroundEffects'

function App() {
  return (
    <div className="app">
      <BackgroundEffects />
      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <Hero />
        <HowItWorks />
        <Features />
        {/*<Metrics />*/}
        <CallToAction />
        <Footer />
      </motion.main>
    </div>
  )
}

export default App
