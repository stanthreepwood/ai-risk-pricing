import { motion } from 'framer-motion'
import styles from './HowItWorks.module.css'

const steps = [
  {
    number: '01',
    title: 'Define Scenarios',
    description: 'Specific AI failure modes with frequency, severity distributions, and propagation characteristics. From historical incident data, regulation frameworks and expert judgement.',
    details: ['Foundation Model Failures', 'Adversarial Attacks', 'Alignment Incidents', 'Regulatory Shocks', 'Supply Chain Compromise'],
  },
  {
    number: '02',
    title: 'Build Dependency Graph',
    description: 'Model the AI supply chain as a directed graph capturing systemic interconnections and concentration risk via HHI index.',
    details: ['Foundation → SaaS → Enterprise', 'Criticality Weighting', 'Concentration Amplification'],
  },
  {
    number: '03',
    title: 'Run Monte Carlo',
    description: 'Execute 100,000+ year simulations generating Year Loss Tables. Poisson frequencies, heavy-tailed severities, correlated propagation.',
    details: ['Pareto & Lognormal Distributions', 'Regime Switching', 'Capability Thresholds'],
  },
  {
    number: '04',
    title: 'Compute Risk Metrics',
    description: 'Calculate standard actuarial measures: Expected Loss, VaR, TVaR. Generate exceedance probability curves and return period tables.',
    details: ['EL, VaR₉₉, TVaR₉₉', 'OEP & AEP Curves', 'Return Period Analysis'],
  },
  {
    number: '05',
    title: 'Pricing AI Safety Mitigation Measures. Discounted for risk reduction.',
    description: 'From the Year Loss Tables, we can price the impact of AI safety mitigation measures. This is done by calculating the risk reduction for each scenario and scenario type, and then discounting the premium for the risk reduction.',
    details: ['Risk Reduction', 'Discounted Premium', 'AI tools vendors'],
  },
  {
    number: '06',
    title: 'Price with Ambiguity',
    description: 'Apply ambiguity loading to compensate for parameter uncertainty. The resulting premium enables informed capital allocation for AI safety.',
    details: ['TVaR Ambiguity Load', 'Expense Loading', 'Rate on Line Calculation'],
  },
]

const HowItWorks = () => {
  return (
    <section className={styles.section} id="how-it-works">
      <div className={styles.container}>
        <motion.div
          className={styles.header}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <span className={styles.label}>Workflow</span>
          <h2 className={styles.title}>
            From Scenarios to
            <br />
            <span className={styles.titleAccent}>Capital Decisions</span>
          </h2>
          <p className={styles.subtitle}>
            A complete catastrophe pricing workflow, from scenario definition to premium calculation.
          </p>
        </motion.div>

        <div className={styles.timeline}>
          {steps.map((step, index) => (
            <motion.div
              key={index}
              className={styles.step}
              initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.15 }}
            >
              <div className={styles.stepConnector}>
                <div className={styles.stepNumber}>{step.number}</div>
                {index < steps.length - 1 && <div className={styles.stepLine} />}
              </div>
              
              <div className={styles.stepContent}>
                <h3 className={styles.stepTitle}>{step.title}</h3>
                <p className={styles.stepDescription}>{step.description}</p>
                <div className={styles.stepDetails}>
                  {step.details.map((detail, i) => (
                    <span key={i} className={styles.stepTag}>{detail}</span>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default HowItWorks
