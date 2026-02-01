import { useEffect, useState } from 'react'
import styles from './BackgroundEffects.module.css'

interface Particle {
  id: number
  x: number
  y: number
  delay: number
  duration: number
  opacity: number
}

const BackgroundEffects = () => {
  const [particles, setParticles] = useState<Particle[]>([])

  useEffect(() => {
    // Generate floating data particles
    const newParticles: Particle[] = []
    for (let i = 0; i < 20; i++) {
      newParticles.push({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        delay: Math.random() * 20,
        duration: 15 + Math.random() * 20,
        opacity: 0.1 + Math.random() * 0.2,
      })
    }
    setParticles(newParticles)
  }, [])

  return (
    <div className={styles.background}>
      {/* Grid pattern */}
      <div className={styles.grid} />
      
      {/* Gradient orbs */}
      <div className={styles.orb1} />
      <div className={styles.orb2} />
      <div className={styles.orb3} />
      
      {/* Floating data particles */}
      <div className={styles.particles}>
        {particles.map((particle) => (
          <div
            key={particle.id}
            className={styles.particle}
            style={{
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              animationDelay: `${particle.delay}s`,
              animationDuration: `${particle.duration}s`,
              opacity: particle.opacity,
            }}
          >
            <span className={styles.particleText}>
              {['λ', 'σ', 'μ', 'α', 'β', 'ρ', 'Σ', '∞', '∂', '∫'][particle.id % 10]}
            </span>
          </div>
        ))}
      </div>
      
      {/* Noise texture overlay */}
      <div className={styles.noise} />
    </div>
  )
}

export default BackgroundEffects
