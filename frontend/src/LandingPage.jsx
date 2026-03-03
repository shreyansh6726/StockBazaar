import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Aurora from './Aurora';

const LandingPage = () => {
    const navigate = useNavigate();
    const [isTransitioning, setIsTransitioning] = useState(false);

    const handleGetStarted = () => {
        setIsTransitioning(true);
        // Allow animation to play before navigating
        setTimeout(() => {
            navigate('/dashboard');
        }, 800);
    };

    const styles = {
        container: {
            minHeight: '100vh',
            backgroundColor: '#050505',
            color: '#f4f4f5',
            fontFamily: "'Inter', sans-serif",
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '20px',
            textAlign: 'center',
            overflow: 'hidden',
            position: 'relative',
        },
        heroSection: {
            zIndex: 1,
            maxWidth: '800px',
            animation: 'fadeIn 1.2s ease-out',
            // Fade out the entire section if transitioning
            transition: 'opacity 0.8s ease-in-out',
            opacity: isTransitioning ? 0 : 1,
        },
        title: {
            fontSize: '4.5rem',
            fontWeight: '900',
            letterSpacing: '0.2em',
            marginBottom: '20px',
            background: 'linear-gradient(to bottom, #ffffff, #a1a1aa)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 0 40px rgba(0,0,0,0.5)',
        },
        subtitle: {
            fontSize: '1.25rem',
            color: '#e4e4e7',
            marginBottom: '40px',
            lineHeight: '1.6',
            maxWidth: '600px',
            margin: '0 auto 40px auto',
            textShadow: '0 2px 10px rgba(0,0,0,0.3)',
        },
        button: {
            padding: '16px 40px',
            fontSize: '1rem',
            fontWeight: '700',
            color: '#000000',
            backgroundColor: '#ffffff',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'transform 0.2s, background-color 0.2s, box-shadow 0.2s, opacity 0.5s',
            boxShadow: '0 0 20px rgba(255, 255, 255, 0.1)',
            position: 'relative',
            zIndex: 2,
        }
    };

    return (
        <div style={styles.container}>
            <style>
                {`
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @keyframes zoomOutFade {
            0% { transform: scale(1); opacity: 1; }
            100% { transform: scale(2.5); opacity: 0; filter: blur(10px); }
          }
          .get-started-btn:hover {
            transform: scale(1.05);
            background-color: #f4f4f5 !important;
            box-shadow: 0 0 30px rgba(255, 255, 255, 0.3) !important;
          }
          .get-started-btn:active {
            transform: scale(0.98);
          }
          .btn-clicked {
            animation: zoomOutFade 0.8s cubic-bezier(0.4, 0, 0.2, 1) forwards !important;
            pointer-events: none;
          }
        `}
            </style>

            <Aurora
                colorStops={["#7cff67", "#B19EEF", "#5227FF"]}
                blend={0.5}
                amplitude={1.0}
                speed={1}
            />

            <div style={styles.heroSection}>
                <h1 style={styles.title}>STOCKBAZAAR</h1>
                <p style={styles.subtitle}>
                    Predict the future of markets with our advanced LSTM neural networks.
                    Real-time analysis, high-confidence insights, and professional-grade tools.
                </p>
                <button
                    className={`get-started-btn ${isTransitioning ? 'btn-clicked' : ''}`}
                    style={styles.button}
                    onClick={handleGetStarted}
                >
                    GET STARTED
                </button>
            </div>

            <div style={{
                position: 'absolute',
                bottom: '40px',
                fontSize: '0.75rem',
                color: '#a1a1aa',
                letterSpacing: '0.1em',
                zIndex: 1,
                textShadow: '0 1px 5px rgba(0,0,0,0.5)',
                transition: 'opacity 0.5s',
                opacity: isTransitioning ? 0 : 1
            }}>
                DEVELOPED BY • SHREYANSH SRIVASTAVA
            </div>
        </div>
    );
};

export default LandingPage;
