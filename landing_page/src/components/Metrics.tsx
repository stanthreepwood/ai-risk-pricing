import { motion, useInView } from 'framer-motion'
import { useRef, useEffect, useState } from 'react'
import styles from './Metrics.module.css'

interface CounterProps {
  end: number
  duration?: number
  prefix?: string
  suffix?: string
  decimals?: number
}

const Counter = ({ end, duration = 2, prefix = '', suffix = '', decimals = 0 }: CounterProps) => {
  const [count, setCount] = useState(0)
  const ref = useRef<HTMLSpanElement>(null)
  const isInView = useInView(ref, { once: true })

  useEffect(() => {
    if (isInView) {
      let startTime: number
      const animate = (currentTime: number) => {
        if (!startTime) startTime = currentTime
        const progress = Math.min((currentTime - startTime) / (duration * 1000), 1)
        const easeProgress = 1 - Math.pow(1 - progress, 3)
        setCount(easeProgress * end)
        if (progress < 1) {
          requestAnimationFrame(animate)
        }
      }
      requestAnimationFrame(animate)
    }
  }, [isInView, end, duration])

  return (
    <span ref={ref}>
      {prefix}{count.toFixed(decimals)}{suffix}
    </span>
  )
}

const metrics = [
  {
    value: 127.45,
    prefix: '$',
    suffix: 'M',
    label: 'Expected Loss',
    description: 'Annual mean loss from AI catastrophe scenarios',
    decimals: 2,
  },
  {
    value: 1.89,
    prefix: '$',
    suffix: 'B',
    label: 'TVaR₉₉',
    description: 'Tail Value at Risk at 99th percentile',
    decimals: 2,
  },
  {
    value: 8.67,
    suffix: 'x',
    label: 'Premium Multiple',
    description: 'Premium to Expected Loss ratio',
    decimals: 2,
  },
  {
    value: 22.6,
    suffix: '%',
    label: 'Rate on Line',
    description: 'Premium as percentage of exposure',
    decimals: 1,
  },
]

const Metrics = () => {
  return (
    <section className={styles.section}>
      <div className={styles.container}>
        <motion.div
          className={styles.header}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <span className={styles.label}>Output</span>
          <h2 className={styles.title}>Risk Metrics That Matter</h2>
          <p className={styles.subtitle}>
            Standard actuarial measures computed from Monte Carlo simulation — 
            the same metrics used in reinsurance treaty pricing.
          </p>
        </motion.div>

        <div className={styles.metricsGrid}>
          {metrics.map((metric, index) => (
            <motion.div
              key={index}
              className={styles.metricCard}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <div className={styles.metricValue}>
                <Counter
                  end={metric.value}
                  prefix={metric.prefix}
                  suffix={metric.suffix}
                  decimals={metric.decimals}
                />
              </div>
              <div className={styles.metricLabel}>{metric.label}</div>
              <div className={styles.metricDescription}>{metric.description}</div>
            </motion.div>
          ))}
        </div>

        <motion.div
          className={styles.formula}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <div className={styles.formulaHeader}>
            <span className={styles.formulaIcon}>∑</span>
            <span className={styles.formulaTitle}>Premium Formula</span>
          </div>
          <div className={styles.formulaContent}>
            <code className={styles.formulaCode}>
              Premium = EL + λ<sub>ambiguity</sub> × TVaR + ε<sub>expense</sub> × EL
            </code>
          </div>
          <p className={styles.formulaNote}>
            The ambiguity load (λ = 50% of TVaR) compensates for parameter uncertainty — 
            critical for AI risks with no historical loss data.
          </p>
        </motion.div>
      </div>
    </section>
  )
}

export default Metrics
