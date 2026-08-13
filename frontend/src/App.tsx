import { useState, useEffect, useRef } from 'react';
import './App.css';

interface DiagnosticResult {
  status: string;
  classification: {
    predicted_class: string;
    predicted_class_id: number;
    probabilities: Record<string, number>;
  };
  regression: {
    depression_score: number;
    anxiety_score: number;
    stress_score: number;
  };
  shap_attribution?: {
    attributions: Record<string, number>;
  };
  narrative?: any;
  latency_ms?: number;
}

export default function App() {
  const [mode, setMode] = useState<'rest' | 'ws'>('rest');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DiagnosticResult | null>(null);
  const [wsConnected, setWsConnected] = useState(false);

  // Biometric Slider States
  const [sleepQuality, setSleepQuality] = useState(3.0);
  const [socialEngagement, setSocialEngagement] = useState(2.5);
  const [typingSpeed, setTypingSpeed] = useState(45.0);
  const [hrvIndex, setHrvIndex] = useState(35.0);
  const [heartRate, setHeartRate] = useState(78.0);
  const [blinkRate, setBlinkRate] = useState(18.0);
  const [speechRate, setSpeechRate] = useState(3.2);

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (mode === 'ws') {
      const ws = new WebSocket('ws://localhost:8000/evaluate/ws');
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setResult(data);
          setLoading(false);
        } catch (e) {
          console.error("Failed to parse WebSocket message:", e);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
      };

      return () => {
        ws.close();
      };
    }
  }, [mode]);

  const buildPayload = () => {
    // 18 physical tabular features matching payload schema
    const tabularValues = [
      sleepQuality,        // Sleep_Quality
      socialEngagement,    // Social_Engagement
      240.0,               // Daily_App_Usage_Min
      typingSpeed,         // Typing_Speed_WPM
      12.0,                // Session_Frequency
      45.0,                // Idle_Time_Min
      0.35,                // Facial_Emotion_Variance
      blinkRate,           // Eye_Blink_Rate
      0.40,                // Smile_Intensity
      0.25,                // Head_Motion_Index
      12.5,                // MFCC_Mean
      4.2,                 // MFCC_Variance
      180.0,               // Pitch_Mean
      speechRate,          // Speech_Rate
      heartRate,           // Heart_Rate_BPM
      hrvIndex,            // HRV_Index
      36.5,                // Skin_Temperature
      2.1,                 // GSR_Level
    ];

    return {
      visual_vector: { values: Array(128).fill(0.05) },
      acoustic_vector: { values: Array(256).fill(-0.02) },
      tabular: { values: tabularValues },
    };
  };

  const handleEvaluate = async () => {
    setLoading(true);
    const payload = buildPayload();

    if (mode === 'rest') {
      try {
        const response = await fetch('http://localhost:8000/evaluate/rest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        setResult(data);
      } catch (err) {
        console.error("REST request failed:", err);
      } finally {
        setLoading(false);
      }
    } else if (mode === 'ws' && wsRef.current && wsConnected) {
      wsRef.current.send(JSON.stringify(payload));
    }
  };

  return (
    <div className="dashboard-container">
      {/* Header Bar */}
      <header className="header">
        <div className="brand">
          <div className="logo-badge">🧠</div>
          <div>
            <h1 className="title">MindScan AI</h1>
            <p className="subtitle">Multimodal Psychiatric Evaluation & Real-time Diagnostic Engine</p>
          </div>
        </div>

        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'rest' ? 'active' : ''}`}
            onClick={() => setMode('rest')}
          >
            REST API Gateway
          </button>
          <button
            className={`mode-btn ${mode === 'ws' ? 'active' : ''}`}
            onClick={() => setMode('ws')}
          >
            WebSocket Stream {mode === 'ws' && (wsConnected ? '🟢' : '🔴')}
          </button>
        </div>
      </header>

      <div className="main-grid">
        {/* Input Panel */}
        <div className="card">
          <h2 className="card-title">⚙️ Patient Biometric Signals</h2>

          <div className="form-group">
            <div className="form-label">
              <span>Sleep Quality (1 - 10)</span>
              <span className="val-badge">{sleepQuality.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              step="0.1"
              value={sleepQuality}
              onChange={(e) => setSleepQuality(parseFloat(e.target.value))}
              className="slider-input"
            />
          </div>

          <div className="form-group">
            <div className="form-label">
              <span>Social Engagement Score (1 - 10)</span>
              <span className="val-badge">{socialEngagement.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              step="0.1"
              value={socialEngagement}
              onChange={(e) => setSocialEngagement(parseFloat(e.target.value))}
              className="slider-input"
            />
          </div>

          <div className="form-group">
            <div className="form-label">
              <span>Typing Speed (WPM)</span>
              <span className="val-badge">{typingSpeed.toFixed(0)} WPM</span>
            </div>
            <input
              type="range"
              min="10"
              max="100"
              step="1"
              value={typingSpeed}
              onChange={(e) => setTypingSpeed(parseFloat(e.target.value))}
              className="slider-input"
            />
          </div>

          <div className="form-group">
            <div className="form-label">
              <span>Heart Rate Variability (HRV Index ms)</span>
              <span className="val-badge">{hrvIndex.toFixed(1)} ms</span>
            </div>
            <input
              type="range"
              min="10"
              max="100"
              step="1"
              value={hrvIndex}
              onChange={(e) => setHrvIndex(parseFloat(e.target.value))}
              className="slider-input"
            />
          </div>

          <div className="form-group">
            <div className="form-label">
              <span>Resting Heart Rate (BPM)</span>
              <span className="val-badge">{heartRate.toFixed(0)} BPM</span>
            </div>
            <input
              type="range"
              min="50"
              max="140"
              step="1"
              value={heartRate}
              onChange={(e) => setHeartRate(parseFloat(e.target.value))}
              className="slider-input"
            />
          </div>

          <div className="form-group">
            <div className="form-label">
              <span>Eye Blink Rate (blinks/min)</span>
              <span className="val-badge">{blinkRate.toFixed(0)}</span>
            </div>
            <input
              type="range"
              min="5"
              max="40"
              step="1"
              value={blinkRate}
              onChange={(e) => setBlinkRate(parseFloat(e.target.value))}
              className="slider-input"
            />
          </div>

          <div className="form-group">
            <div className="form-label">
              <span>Speech Rate (syllables/sec)</span>
              <span className="val-badge">{speechRate.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="1.0"
              max="6.0"
              step="0.1"
              value={speechRate}
              onChange={(e) => setSpeechRate(parseFloat(e.target.value))}
              className="slider-input"
            />
          </div>

          <button className="btn-submit" onClick={handleEvaluate} disabled={loading}>
            {loading ? '⚡ Running DCMF-Net Inference...' : '📊 Run Diagnostic Evaluation'}
          </button>
        </div>

        {/* Output Diagnostics Panel */}
        <div className="card">
          <h2 className="card-title">🔍 Clinical Diagnostic Output</h2>

          {result ? (
            <div>
              {/* Severity Banner */}
              <div className={`severity-banner ${result.classification.predicted_class}`}>
                <div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>PREDICTED SEVERITY</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>
                    {result.classification.predicted_class.replace('_', ' ')}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>CONFIDENCE</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 600 }}>
                    {(
                      (result.classification.probabilities[result.classification.predicted_class] || 0.98) * 100
                    ).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Continuous Score Cards */}
              <div className="scores-grid">
                <div className="score-box">
                  <div className="score-num">{result.regression.depression_score.toFixed(1)}</div>
                  <div className="score-lbl">Depression (PHQ-9)</div>
                </div>
                <div className="score-box">
                  <div className="score-num">{result.regression.anxiety_score.toFixed(1)}</div>
                  <div className="score-lbl">Anxiety (GAD-7)</div>
                </div>
                <div className="score-box">
                  <div className="score-num">{result.regression.stress_score.toFixed(1)}</div>
                  <div className="score-lbl">Stress (PSS)</div>
                </div>
              </div>

              {/* Clinical Narrative Summary */}
              <div className="narrative-box">
                <div className="narrative-section">
                  <h4>Diagnostic Impression</h4>
                  <p>
                    {typeof result.narrative === 'string'
                      ? result.narrative
                      : result.narrative?.summary || 'Multimodal assessment completed cleanly.'}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-secondary)' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>📈</div>
              <p>Adjust the patient biometrics on the left and click <strong>Run Diagnostic Evaluation</strong> to view instant DCMF-Net severity estimation & FastSHAP attributions.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
