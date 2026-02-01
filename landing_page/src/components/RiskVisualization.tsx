import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import styles from './RiskVisualization.module.css'

interface DataPoint {
  x: number
  y: number
}

const generateExceedanceCurve = (): DataPoint[] => {
  const points: DataPoint[] = []
  for (let i = 0; i <= 100; i += 2) {
    const x = i / 100
    // Heavy-tailed exceedance curve (Pareto-like)
    const y = Math.pow(1 - x * 0.99, 1.5)
    points.push({ x: x * 300, y: (1 - y) * 200 })
  }
  return points
}

const RiskVisualization = () => {
  const [curvePoints] = useState<DataPoint[]>(generateExceedanceCurve())
  const [activeMetric, setActiveMetric] = useState(0)
  
  const metrics = [
    { label: 'EL', value: '$127.5M', desc: 'Expected Loss' },
    { label: 'VaR₉₉', value: '$1.25B', desc: 'Value at Risk' },
    { label: 'TVaR₉₉', value: '$1.89B', desc: 'Tail Value at Risk' },
  ]
  
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveMetric((prev) => (prev + 1) % metrics.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [])
  
  const pathD = curvePoints
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
    .join(' ')
  
  return (
    <div className={styles.container}>
      {/* Glow effect */}
      <div className={styles.glow} />
      
      {/* Main visualization card */}
      <motion.div 
        className={styles.card}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <div className={styles.cardHeader}>
          <span className={styles.cardTitle}>Exceedance Probability Curve</span>
          <div className={styles.cardStatus}>
            <span className={styles.statusDot} />
            <span>Live Simulation</span>
          </div>
        </div>
        
        <div className={styles.chart}>
          <svg viewBox="0 0 320 220" className={styles.svg}>
            {/* Grid lines */}
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--color-border-subtle)" strokeWidth="0.5" />
              </pattern>
              <linearGradient id="curveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="var(--color-accent-gold)" />
                <stop offset="100%" stopColor="var(--color-accent-gold-dark)" />
              </linearGradient>
              <linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="var(--color-accent-gold)" stopOpacity="0.3" />
                <stop offset="100%" stopColor="var(--color-accent-gold)" stopOpacity="0" />
              </linearGradient>
            </defs>
            
            <rect width="300" height="200" x="10" y="10" fill="url(#grid)" />
            
            {/* Area under curve */}
            <motion.path
              d={`${pathD} L 300 200 L 0 200 Z`}
              fill="url(#areaGradient)"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1, delay: 0.5 }}
            />
            
            {/* Main curve */}
            <motion.path
              d={pathD}
              fill="none"
              stroke="url(#curveGradient)"
              strokeWidth="2.5"
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1.5, ease: 'easeOut' }}
            />
            
            {/* VaR marker */}
            <motion.g
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.2 }}
            >
              <line x1="240" y1="10" x2="240" y2="200" stroke="var(--color-risk-high)" strokeWidth="1" strokeDasharray="4 4" />
              <circle cx="240" cy="175" r="4" fill="var(--color-risk-high)" />
              <text x="250" y="30" fill="var(--color-text-muted)" fontSize="10" fontFamily="var(--font-mono)">VaR₉₉</text>
            </motion.g>
            
            {/* Axis labels */}
            <text x="160" y="218" fill="var(--color-text-muted)" fontSize="9" textAnchor="middle">Loss Amount ($M)</text>
            <text x="8" y="110" fill="var(--color-text-muted)" fontSize="9" textAnchor="middle" transform="rotate(-90, 8, 110)">Exceedance Prob.</text>
          </svg>
        </div>
        
        {/* Metrics row */}
        <div className={styles.metrics}>
          {metrics.map((metric, i) => (
            <motion.div
              key={metric.label}
              className={`${styles.metric} ${activeMetric === i ? styles.metricActive : ''}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + i * 0.1 }}
            >
              <span className={styles.metricLabel}>{metric.label}</span>
              <span className={styles.metricValue}>{metric.value}</span>
              <span className={styles.metricDesc}>{metric.desc}</span>
            </motion.div>
          ))}
        </div>
      </motion.div>
      
      {/* Floating scenario cards */}
      <motion.div
        className={`${styles.floatingCard} ${styles.floatingCard1}`}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.8 }}
      >
        <span className={styles.floatingIcon}>⚡</span>
        <div>
          <span className={styles.floatingTitle}>Foundation Model</span>
          <span className={styles.floatingValue}>λ = 0.15</span>
        </div>
      </motion.div>
      
      <motion.div
        className={`${styles.floatingCard} ${styles.floatingCard2}`}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 1 }}
      >
        <span className={styles.floatingIcon}>🔗</span>
        <div>
          <span className={styles.floatingTitle}>Supply Chain</span>
          <span className={styles.floatingValue}>HHI: 0.42</span>
        </div>
      </motion.div>
      
      <motion.div
        className={`${styles.floatingCard} ${styles.floatingCard3}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.2 }}
      >
        <span className={styles.floatingIcon}>📊</span>
        <div>
          <span className={styles.floatingTitle}>Correlation</span>
          <span className={styles.floatingValue}>ρ = 0.65</span>
        </div>
      </motion.div>
    </div>
  )
}

export default RiskVisualization
