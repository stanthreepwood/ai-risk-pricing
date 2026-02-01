import { motion } from 'framer-motion'
import styles from './Features.module.css'

const features = [
  {
    icon: '◈',
    title: 'Scenario-Based Modeling',
    description: 'Five distinct AI failure modes: Foundation Model Failure, Adversarial Attacks, Alignment Failures, Regulatory Shocks, and Supply Chain Compromises.',
    highlight: 'No historical data required',
  },
  {
    icon: '∿',
    title: 'Heavy-Tailed Distributions',
    description: 'Pareto and Lognormal severity distributions capture extreme tail risk inherent in catastrophic AI events.',
    highlight: 'Fat-tail ready',
  },
  {
    icon: '⟁',
    title: 'Dependency Graph',
    description: 'NetworkX-powered supply chain modeling tracks loss propagation through Foundation Models → SaaS → Enterprises.',
    highlight: 'Systemic risk capture',
  },
  {
    icon: '∞',
    title: 'Monte Carlo Engine',
    description: '100,000+ year simulations generate Year Loss Tables for robust statistical inference on rare events.',
    highlight: 'High-performance',
  },
  {
    icon: '⊞',
    title: 'Correlation Modeling',
    description: 'Concentration amplification via HHI index and criticality weighting for correlated loss scenarios.',
    highlight: 'Dependency-aware',
  },
  {
    icon: '⟐',
    title: 'Ambiguity Loading',
    description: 'Premium formula includes uncertainty margins, compensating for parameter uncertainty in novel risk classes.',
    highlight: 'Prudent pricing',
  },
]

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
}

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
}

const Features = () => {
  return (
    <section className={styles.section} id="features">
      <div className={styles.container}>
        <motion.div
          className={styles.header}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <span className={styles.label}>Capabilities</span>
          <h2 className={styles.title}>
            Actuarial-Grade
            <br />
            <span className={styles.titleAccent}>Risk Engineering</span>
          </h2>
          <p className={styles.subtitle}>
            Built on the same methodologies used by reinsurance analytics teams for natural catastrophe modeling.
          </p>
        </motion.div>

        <motion.div
          className={styles.grid}
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              className={styles.card}
              variants={item}
              whileHover={{ 
                y: -4, 
                borderColor: 'var(--color-accent-gold-dark)',
                transition: { duration: 0.2 } 
              }}
            >
              <div className={styles.cardIcon}>{feature.icon}</div>
              <h3 className={styles.cardTitle}>{feature.title}</h3>
              <p className={styles.cardDescription}>{feature.description}</p>
              <span className={styles.cardHighlight}>{feature.highlight}</span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}

export default Features
