import { motion } from 'framer-motion'
import styles from './CallToAction.module.css'

const CallToAction = () => {
  return (
    <section className={styles.section}>
      <div className={styles.container}>
        <motion.div
          className={styles.content}
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className={styles.glow} />
          
          <span className={styles.preheader}>Unlock AI Safety Investments</span>
          
          <h2 className={styles.title}>
            Quantify the
            <br />
            <span className={styles.titleAccent}>Unquantifiable</span>
          </h2>
          
          <p className={styles.description}>
            Transform AI catastrophe risk from an abstract concern into actionable capital allocation decisions.
            Model correlations, capture tail risk, and enable informed safety investments.
          </p>
          
          <div className={styles.cta}>
            <motion.a
              href="/app"
              className={styles.ctaPrimary}
              whileHover={{ scale: 1.02, boxShadow: '0 0 60px rgba(201, 169, 98, 0.5)' }}
              whileTap={{ scale: 0.98 }}
            >
              <span>Launch Risk Engine</span>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </motion.a>
          </div>
          
          <div className={styles.features}>
            <div className={styles.feature}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
              <span>Open Source</span>
            </div>
            <div className={styles.feature}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
              <span>No Data Required</span>
            </div>
            <div className={styles.feature}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
              <span>Actuarial Standards</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

export default CallToAction
