import { motion } from 'framer-motion'
import styles from './Hero.module.css'
import RiskVisualization from './RiskVisualization'

const Hero = () => {
  return (
    <section className={styles.hero}>
      <div className={styles.container}>
        <motion.div
          className={styles.content}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        >
          <div className={styles.badge}>
            <img 
              src="https://framerusercontent.com/images/UHxUj0LmVxOq70NyvUz698P5IY.png?width=174&height=164" 
              alt="Badge" 
              style={{
                height: '1.5em',
                width: 'auto',
                verticalAlign: 'middle',
                marginRight: '0.5em',
                display: 'inline-block'
              }}
            />
            <span>Technical AI governance for Apart Hackathon</span>
          </div>
          
          <h1 className={styles.title}>
            <span className={styles.titleLine}>Quantify</span>
            <span className={`${styles.titleLine} ${styles.titleAccent}`}>AI Risk</span>
          </h1>
          
          <p className={styles.subtitle}>
            Actuarial-grade catastrophe modeling for AI systems.
            <br />
            <span className={styles.subtitleHighlight}>
              Correlations. Tail Risk. Capital Allocation.
            </span>
          </p>
          
          <p className={styles.description}>
            Model systemic AI failures with Monte Carlo simulation, heavy-tailed distributions, 
            and dependency graphs — unlocking informed capital allocation for AI safety investments.
          </p>
          
          <div className={styles.cta}>
            <motion.a
              href="https://pricing-ai-risks.streamlit.app/"
              className={styles.ctaPrimary}
              whileHover={{ scale: 1.02, boxShadow: '0 0 40px rgba(201, 169, 98, 0.4)' }}
              whileTap={{ scale: 0.98 }}
            >
              <span>Launch Risk Engine</span>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </motion.a>
            
            <motion.a
              href="https://github.com/stanthreepwood/ai-risk-pricing"
              className={styles.ctaSecondary}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
              </svg>
              <span>View Source</span>
            </motion.a>
          </div>
          
          <div className={styles.stats}>
            <div className={styles.stat}>
              <span className={styles.statValue}>100K+</span>
              <span className={styles.statLabel}>Simulated Years</span>
            </div>
            <div className={styles.statDivider} />
            <div className={styles.stat}>
              <span className={styles.statValue}>TVaR<sub>99</sub></span>
              <span className={styles.statLabel}>Tail Risk Metric</span>
            </div>
            <div className={styles.statDivider} />
            <div className={styles.stat}>
              <span className={styles.statValue}>5</span>
              <span className={styles.statLabel}>Scenario Types</span>
            </div>
          </div>
        </motion.div>
        
        <motion.div
          className={styles.visualization}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2, ease: 'easeOut' }}
        >
          <RiskVisualization />
        </motion.div>
      </div>
      
      <div className={styles.scrollIndicator}>
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 5v14M5 12l7 7 7-7" />
          </svg>
        </motion.div>
      </div>
    </section>
  )
}

export default Hero
