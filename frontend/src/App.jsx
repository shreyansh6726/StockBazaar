import React, { useState } from 'react';
import axios from 'axios';

const BACKEND_URL = "https://stockbazaar.onrender.com";

const App = () => {
  const [symbol, setSymbol] = useState('');
  const [tenure, setTenure] = useState('1h');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const getPrediction = async () => {
    if (!symbol) return;
    setLoading(true);
    try {
      const response = await axios.get(`${BACKEND_URL}/predict?symbol=${symbol}&tenure=${tenure}`);
      setResult(response.data);
    } catch (error) {
      console.error("Error:", error);
      alert("Backend is initializing. Please wait a moment.");
    }
    setLoading(false);
  };

  // --- Styles Objects ---
  const styles = {
    container: {
      minHeight: '100vh',
      backgroundColor: '#0a0a0c',
      color: '#f4f4f5',
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '40px 20px',
    },
    header: {
      fontSize: '2.5rem',
      fontWeight: '800',
      letterSpacing: '-0.05em',
      background: 'linear-gradient(to right, #ffffff, #a1a1aa)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      marginBottom: '50px',
      animation: 'fadeIn 1s ease-out',
    },
    inputCard: {
      background: '#18181b',
      padding: '30px',
      borderRadius: '16px',
      border: '1px solid #27272a',
      display: 'flex',
      gap: '15px',
      width: '100%',
      maxWidth: '700px',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
      marginBottom: '40px',
    },
    input: {
      flex: 2,
      background: '#09090b',
      border: '1px solid #3f3f46',
      borderRadius: '8px',
      padding: '12px 16px',
      color: 'white',
      fontSize: '1rem',
      outline: 'none',
      transition: 'border-color 0.2s',
    },
    select: {
      flex: 1,
      background: '#09090b',
      border: '1px solid #3f3f46',
      borderRadius: '8px',
      padding: '12px',
      color: 'white',
      cursor: 'pointer',
    },
    button: {
      flex: 1,
      background: loading ? '#27272a' : '#ffffff',
      color: '#000000',
      border: 'none',
      borderRadius: '8px',
      fontWeight: '600',
      cursor: loading ? 'not-allowed' : 'pointer',
      transition: 'transform 0.2s, opacity 0.2s',
    },
    resultCard: {
      width: '100%',
      maxWidth: '700px',
      background: 'linear-gradient(145deg, #18181b 0%, #09090b 100%)',
      borderRadius: '20px',
      padding: '40px',
      border: '1px solid #27272a',
      textAlign: 'center',
      animation: 'slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
    },
    price: {
      fontSize: '4rem',
      fontWeight: '900',
      margin: '10px 0',
      color: '#ffffff',
    },
    trendBadge: (trend) => ({
      display: 'inline-block',
      padding: '6px 16px',
      borderRadius: '100px',
      fontSize: '0.85rem',
      fontWeight: '700',
      backgroundColor: trend === 'UP' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
      color: trend === 'UP' ? '#4ade80' : '#f87171',
      border: `1px solid ${trend === 'UP' ? '#166534' : '#7f1d1d'}`,
      marginBottom: '20px',
    }),
    footerText: {
      color: '#71717a',
      fontSize: '0.9rem',
      marginTop: '15px'
    }
  };

  return (
    <div style={styles.container}>
      {/* Dynamic Keyframes injected into a style tag */}
      <style>
        {`
          @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
          @keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
          input:focus { border-color: #ffffff !important; }
        `}
      </style>

      <h1 style={styles.header}>STOCKBAZAAR</h1>

      <div style={styles.inputCard}>
        <input 
          style={styles.input}
          placeholder="Ticker Symbol (e.g. NVDA)"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
        />
        <select 
          style={styles.select}
          value={tenure}
          onChange={(e) => setTenure(e.target.value)}
        >
          <option value="1h">1 Hour</option>
          <option value="1d">1 Day</option>
          <option value="1w">1 Week</option>
        </select>
        <button 
          style={styles.button}
          onClick={getPrediction}
          disabled={loading}
          onMouseEnter={(e) => e.target.style.opacity = '0.9'}
          onMouseLeave={(e) => e.target.style.opacity = '1'}
        >
          {loading ? "ANALYZING..." : "PREDICT"}
        </button>
      </div>

      {result && (
        <div style={styles.resultCard}>
          <div style={styles.trendBadge(result.prediction.trend)}>
            {result.prediction.trend} TREND DETECTED
          </div>
          <div style={{color: '#a1a1aa', fontSize: '0.9rem'}}>Upcoming Predicted Price</div>
          <div style={styles.price}>${result.prediction.predicted_price}</div>
          <div style={styles.footerText}>
            Current Reference: ${result.prediction.last_close} • {result.metadata.tenure} Horizon
          </div>
        </div>
      )}
    </div>
  );
};

export default App;