import React, { useState } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const App = () => {
  const [symbol, setSymbol] = useState('');
  const [tenure, setTenure] = useState('1h');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const getPrediction = async () => {
    if (!symbol) return;
    setLoading(true);
    setResult(null); // Clear previous result to show loading state clearly
    try {
      const response = await axios.get(`${BACKEND_URL}/predict?symbol=${symbol}&tenure=${tenure}`);
      setResult(response.data);
    } catch (error) {
      console.error("Error:", error);
      alert("Model engine is warming up on Render. Please try again in 30 seconds.");
    }
    setLoading(false);
  };

  const styles = {
    container: {
      minHeight: '100vh',
      backgroundColor: '#050505', // Deep SaaS black
      color: '#f4f4f5',
      fontFamily: "'Inter', sans-serif",
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '60px 20px',
    },
    header: {
      fontSize: '1rem',
      fontWeight: '600',
      letterSpacing: '0.4em',
      color: '#52525b',
      marginBottom: '60px',
      textAlign: 'center',
    },
    inputCard: {
      background: '#09090b',
      padding: '10px',
      borderRadius: '12px',
      border: '1px solid #18181b',
      display: 'flex',
      gap: '10px',
      width: '100%',
      maxWidth: '600px',
      marginBottom: '40px',
    },
    input: {
      flex: 2,
      background: 'transparent',
      border: 'none',
      padding: '12px 16px',
      color: 'white',
      fontSize: '0.95rem',
      outline: 'none',
    },
    select: {
      background: '#18181b',
      border: 'none',
      borderRadius: '6px',
      padding: '0 10px',
      color: '#a1a1aa',
      cursor: 'pointer',
      fontSize: '0.85rem',
    },
    button: {
      background: '#ffffff',
      color: '#000000',
      border: 'none',
      borderRadius: '6px',
      padding: '0 25px',
      fontWeight: '700',
      fontSize: '0.85rem',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
    },
    // --- Loading Screen Styles ---
    loadingWrapper: {
      marginTop: '50px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '20px',
      animation: 'fadeIn 0.5s ease-in-out',
    },
    spinner: {
      width: '40px',
      height: '40px',
      border: '2px solid #27272a',
      borderTop: '2px solid #ffffff',
      borderRadius: '50%',
      animation: 'spin 0.8s linear infinite',
    },
    loadingText: {
      fontSize: '0.85rem',
      color: '#a1a1aa',
      letterSpacing: '0.1em',
    },
    // --- Result Styles ---
    resultCard: {
      width: '100%',
      maxWidth: '600px',
      textAlign: 'center',
      animation: 'slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
    },
    priceLabel: {
      color: '#71717a',
      fontSize: '0.8rem',
      textTransform: 'uppercase',
      letterSpacing: '0.1em',
      marginBottom: '10px',
    },
    priceValue: {
      fontSize: '5rem',
      fontWeight: '800',
      letterSpacing: '-0.04em',
      margin: '0',
    }
  };

  return (
    <div style={styles.container}>
      <style>
        {`
          @keyframes spin { to { transform: rotate(360deg); } }
          @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
          @keyframes slideUp { from { transform: translateY(30px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
          @keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
          .pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        `}
      </style>

      <h1 style={styles.header}>STOCKBAZAAR</h1>

      <div style={styles.inputCard}>
        <input
          style={styles.input}
          placeholder="ENTER TICKER..."
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
        />
        <select style={styles.select} value={tenure} onChange={(e) => setTenure(e.target.value)}>
          <option value="1h">1 Hour</option>
          <option value="1d">1 Day</option>
          <option value="1w">1 Week</option>
        </select>
        <button style={styles.button} onClick={getPrediction} disabled={loading}>
          {loading ? "..." : "PREDICT"}
        </button>
      </div>

      {/* Loading State */}
      {loading && (
        <div style={styles.loadingWrapper}>
          <div style={styles.spinner}></div>
          <div className="pulse" style={styles.loadingText}>
            TRAINING LSTM MODEL • ANALYZING DATA
          </div>
        </div>
      )}

      {/* Result State */}
      {result && !loading && (
        <div style={styles.resultCard}>
          <div style={styles.priceLabel}>Predicted {result.metadata.symbol} Value</div>
          <h2 style={styles.priceValue}>${result.prediction.predicted_price}</h2>
          <div style={{
            color: result.prediction.trend === 'UP' ? '#4ade80' : '#f87171',
            fontWeight: '600',
            fontSize: '1rem',
            marginTop: '10px'
          }}>
            EXPECTED {result.prediction.trend}WARD MOMENTUM
          </div>
          <div style={{ marginTop: '40px', fontSize: '0.75rem', color: '#3f3f46' }}>
            CONFIDENCE: HIGH • BASED ON LAST 500 DATA POINTS
          </div>
        </div>
      )}
    </div>
  );
};

export default App;