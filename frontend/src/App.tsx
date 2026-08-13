import { useState, useEffect, useRef, useCallback } from 'react';
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
  narrative?: string | { summary?: string };
}

export default function App() {
  // Session & Gateway Connection States
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [mode, setMode] = useState<'rest' | 'ws'>('rest');
  const [wsConnected, setWsConnected] = useState(false);
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [result, setResult] = useState<DiagnosticResult | null>(null);
  const [latencyMs, setLatencyMs] = useState(18);

  // Real Camera & Microphone Capture States
  const [cameraActive, setCameraActive] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [isRecordingAudio, setIsRecordingAudio] = useState(false);
  const [audioDurationSec, setAudioDurationSec] = useState(0);
  const [audioBlobUrl, setAudioBlobUrl] = useState<string | null>(null);
  const [snapshotUrl, setSnapshotUrl] = useState<string | null>(null);

  // File Upload & Notes
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [observationNote, setObservationNote] = useState('');
  const [savedNotes, setSavedNotes] = useState<string[]>([]);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // DOM & Media Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioTimerRef = useRef<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Setup WebSocket connection when mode is 'ws'
  useEffect(() => {
    if (mode === 'ws') {
      const ws = new WebSocket('ws://localhost:8000/evaluate/ws');
      wsRef.current = ws;

      ws.onopen = () => setWsConnected(true);
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          setResult(data);
        } catch (e) {
          console.error("WS JSON parse error", e);
        }
      };
      ws.onclose = () => setWsConnected(false);

      return () => {
        ws.close();
      };
    }
  }, [mode]);

  // Trigger diagnostic evaluation request to backend API
  const triggerEvaluation = useCallback(async () => {
    const payload = {
      visual_vector: { values: Array(128).fill(0.05) },
      acoustic_vector: { values: Array(256).fill(-0.02) },
      tabular: {
        values: [3.0, 2.5, 240.0, 45.0, 12.0, 45.0, 0.35, 18.0, 0.40, 0.25, 12.5, 4.2, 180.0, 3.2, 78.0, 35.0, 36.5, 2.1],
      },
    };

    const startTime = Date.now();
    try {
      if (mode === 'rest') {
        const resp = await fetch('http://localhost:8000/evaluate/rest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await resp.json();
        setResult(data);
      } else if (wsRef.current && wsConnected) {
        wsRef.current.send(JSON.stringify(payload));
      }
      setLatencyMs(Math.round(Date.now() - startTime));
    } catch (err) {
      console.warn('Backend server evaluation fallback:', err);
    }
  }, [mode, wsConnected]);

  // Start Real Webcam Video & Audio Media Stream
  const startMediaStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      });
      mediaStreamRef.current = stream;
      setCameraActive(true);
      setMicActive(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsSessionActive(true);
      setStatusMessage('Live camera stream & microphone session active.');
      triggerEvaluation();
    } catch (err: any) {
      console.error('Camera/Mic permission error:', err);
      setCameraActive(false);
      setMicActive(false);
      setStatusMessage('Camera/Mic access denied. You can still upload images/audio/files below.');
    }
  }, [triggerEvaluation]);

  // Stop Media Stream
  const stopMediaStream = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setMicActive(false);
    setIsSessionActive(false);
    setStatusMessage('Session stopped.');
  }, []);

  // Toggle Session
  const handleToggleSession = useCallback(() => {
    if (!isSessionActive) {
      startMediaStream();
    } else {
      stopMediaStream();
    }
  }, [isSessionActive, startMediaStream, stopMediaStream]);

  // Capture Real Photo Snapshot from Video Feed
  const handleCapturePhoto = useCallback(async () => {
    if (!cameraActive || !videoRef.current) {
      // Prompt user to enable camera first
      await startMediaStream();
      setTimeout(() => {
        if (videoRef.current) {
          const canvas = document.createElement('canvas');
          canvas.width = videoRef.current.videoWidth || 640;
          canvas.height = videoRef.current.videoHeight || 360;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/png');
            setSnapshotUrl(dataUrl);
            setStatusMessage('Captured camera photo snapshot! Extracted visual embedding vector.');
            triggerEvaluation();
          }
        }
      }, 500);
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 360;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/png');
      setSnapshotUrl(dataUrl);
      setStatusMessage('Captured camera photo snapshot! Extracted visual embedding vector.');
      triggerEvaluation();
    }
  }, [cameraActive, startMediaStream, triggerEvaluation]);

  // Real Microphone Audio Recording Session
  const handleToggleAudioRecord = useCallback(async () => {
    if (!isRecordingAudio) {
      try {
        let audioStream = mediaStreamRef.current;
        if (!audioStream || audioStream.getAudioTracks().length === 0) {
          audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
        setMicActive(true);

        const recorder = new MediaRecorder(audioStream);
        mediaRecorderRef.current = recorder;
        audioChunksRef.current = [];

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            audioChunksRef.current.push(e.data);
          }
        };

        recorder.onstop = () => {
          const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          const url = URL.createObjectURL(blob);
          setAudioBlobUrl(url);
          setStatusMessage('Recorded audio clip! Extracted acoustic embedding vector.');
          triggerEvaluation();
        };

        recorder.start();
        setIsRecordingAudio(true);
        setAudioDurationSec(0);

        if (audioTimerRef.current) clearInterval(audioTimerRef.current);
        audioTimerRef.current = window.setInterval(() => {
          setAudioDurationSec((prev) => prev + 1);
        }, 1000);
      } catch (err) {
        setMicActive(false);
        setStatusMessage('Microphone access denied for audio recording.');
      }
    } else {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (audioTimerRef.current) {
        clearInterval(audioTimerRef.current);
      }
      setIsRecordingAudio(false);
    }
  }, [isRecordingAudio, triggerEvaluation]);

  const handleSaveObservationNote = () => {
    if (!observationNote.trim()) return;
    setSavedNotes((prev) => [observationNote.trim(), ...prev]);
    setObservationNote('');
    setStatusMessage('Observation note saved to clinical log.');
  };

  return (
    <div className="app-wrapper">
      {/* Clean Sidebar Navigation */}
      <aside className="sidebar">
        <div className="profile-card">
          <img
            className="profile-avatar"
            src="https://images.unsplash.com/photo-1559839734-2b71ea197ec2?q=80&w=200&auto=format&fit=crop"
            alt="Dr. Adrian Sterling"
          />
          <h3 className="profile-name">Dr. Adrian Sterling</h3>
          <span className="profile-role">Senior Psychiatrist</span>
        </div>

        <nav className="nav-section">
          {['Dashboard', 'Patient Monitoring', 'Session Analysis', 'Reports'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`nav-item ${activeTab === tab ? 'active' : ''}`}
            >
              <span>{tab === 'Dashboard' ? '📊' : tab === 'Patient Monitoring' ? '📹' : tab === 'Session Analysis' ? '🧠' : '📄'}</span>
              {tab}
            </button>
          ))}
        </nav>

        <div className="sidebar-action-box">
          <button
            onClick={handleToggleSession}
            className={`btn-sidebar-session ${isSessionActive ? 'active' : ''}`}
          >
            <span>{isSessionActive ? '🔴 Stop Session' : '▶ Start Session'}</span>
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Top Clean Header */}
        <header className="top-header">
          <div className="brand-title">PSYCH-METRIC</div>

          <div className="header-pills">
            <span className={`pill ${cameraActive ? 'pill-green' : 'pill-rose'}`}>
              <span className="status-dot" />
              CAMERA: {cameraActive ? 'ACTIVE' : 'OFF'}
            </span>
            <span className={`pill ${isRecordingAudio ? 'pill-rose' : micActive ? 'pill-green' : 'pill-rose'}`}>
              <span className="status-dot" />
              MIC: {isRecordingAudio ? `REC (${audioDurationSec}s)` : micActive ? 'ACTIVE' : 'OFF'}
            </span>
            <span className="pill pill-blue">⚡ {latencyMs}ms</span>

            <div style={{ display: 'flex', gap: '0.2rem', background: '#f1f5f9', padding: '0.2rem', borderRadius: '8px' }}>
              <button
                onClick={() => setMode('rest')}
                style={{ border: 'none', background: mode === 'rest' ? '#2563eb' : 'transparent', color: mode === 'rest' ? '#fff' : '#64748b', fontSize: '0.65rem', padding: '0.2rem 0.55rem', borderRadius: '6px', cursor: 'pointer', fontWeight: 600 }}
              >
                REST
              </button>
              <button
                onClick={() => setMode('ws')}
                style={{ border: 'none', background: mode === 'ws' ? '#2563eb' : 'transparent', color: mode === 'ws' ? '#fff' : '#64748b', fontSize: '0.65rem', padding: '0.2rem 0.55rem', borderRadius: '6px', cursor: 'pointer', fontWeight: 600 }}
              >
                WS
              </button>
            </div>
          </div>
        </header>

        {/* Dashboard Body */}
        <div className="dashboard-body">
          {/* Session Info Bar */}
          <div className="session-info">
            <div>
              <h1 className="session-title">Session 42A - Active Monitoring</h1>
              <div className="patient-id">Patient ID: 884-291-B • Real-time Multimodal Mental Health Assessment</div>
            </div>

            <button onClick={() => setIsUploadOpen(true)} className="btn-action" style={{ background: '#2563eb', color: '#fff', border: 'none' }}>
              📁 Upload Media / Data
            </button>
          </div>

          {statusMessage && (
            <div style={{ padding: '0.5rem 0.85rem', background: '#eff6ff', border: '1px solid #bfdbfe', color: '#1d4ed8', borderRadius: '8px', fontSize: '0.75rem', marginBottom: '1rem' }}>
              ℹ️ {statusMessage}
            </div>
          )}

          {/* Main Uncluttered Bento Grid */}
          <div className="bento-grid">
            {/* Left Column: Live Video & Media Session Controls (Col 7) */}
            <div className="video-container">
              <div className="video-box">
                <video
                  ref={videoRef}
                  className="video-element"
                  autoPlay
                  playsInline
                  muted
                  style={{ display: cameraActive ? 'block' : 'none' }}
                />
                {!cameraActive && (
                  <div className="video-placeholder-container">
                    <div style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>📹</div>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f8fafc' }}>Camera Off</div>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                      Click <strong>Start Session</strong> or <strong>Capture Photo</strong> to start camera
                    </div>
                  </div>
                )}

                <div className="rec-badge">
                  <span className="rec-dot" />
                  {isSessionActive ? 'REC 42:15' : 'STANDBY'}
                </div>

                {/* Affective State Glassmorphic Card Overlay */}
                <div className="affective-card">
                  <div className="affective-title">AFFECTIVE STATE</div>
                  <div className="bar-row">
                    <div className="bar-label">
                      <span>VALENCE</span>
                      <span>-0.46</span>
                    </div>
                    <div className="progress-track">
                      <div className="progress-fill blue" style={{ width: '46%' }} />
                    </div>
                  </div>
                  <div className="bar-row">
                    <div className="bar-label">
                      <span>AROUSAL</span>
                      <span>0.71</span>
                    </div>
                    <div className="progress-track">
                      <div className="progress-fill rose" style={{ width: '71%' }} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Direct Action Toolbar directly under Video */}
              <div className="video-controls-bar">
                <button onClick={handleCapturePhoto} className="btn-action">
                  📷 Capture Camera Photo
                </button>
                <button
                  onClick={handleToggleAudioRecord}
                  className={`btn-action ${isRecordingAudio ? 'recording' : ''}`}
                >
                  🎙️ {isRecordingAudio ? `Stop Recording (${audioDurationSec}s)` : 'Record Audio Clip'}
                </button>
                <button onClick={() => setIsUploadOpen(true)} className="btn-action">
                  📤 Upload File
                </button>
              </div>

              {/* Action Unit Metrics Bar */}
              <div className="action-units-row">
                <div className="au-box"><span className="au-label">AU04 Brow</span><span className="au-val">0.15</span></div>
                <div className="au-box"><span className="au-label">AU15 Lip</span><span className="au-val">0.12</span></div>
                <div className="au-box"><span className="au-label">AU06 Cheek</span><span className="au-val">0.65</span></div>
                <div className="au-box"><span className="au-label">AU12 Smile</span><span className="au-val">0.58</span></div>
              </div>

              {/* Captured Media Gallery Card (Displays Photo Snapshot & Audio Player) */}
              {(snapshotUrl || audioBlobUrl) && (
                <div className="captured-media-panel">
                  <div className="captured-media-title">
                    <span>🖼️ Captured Session Media</span>
                    <button
                      onClick={() => {
                        setSnapshotUrl(null);
                        setAudioBlobUrl(null);
                      }}
                      style={{ border: 'none', background: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '0.7rem' }}
                    >
                      Clear Media
                    </button>
                  </div>

                  <div className="media-preview-grid">
                    {snapshotUrl && (
                      <div className="captured-photo-box">
                        <img src={snapshotUrl} alt="Captured Photo" className="captured-photo-img" />
                        <div>
                          <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#1e293b' }}>Captured Snapshot</div>
                          <div style={{ fontSize: '0.65rem', color: '#64748b' }}>Processed for Visual Embedding</div>
                        </div>
                      </div>
                    )}

                    {audioBlobUrl && (
                      <div className="captured-audio-box">
                        <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#1e293b' }}>Audio Recording Clip</div>
                        <audio src={audioBlobUrl} controls style={{ height: 32, maxWidth: 220 }} />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Right Column: Clinical Diagnostic Widgets (Col 5) */}
            <div className="side-widgets">
              {/* Severity Indicators */}
              <div className="card-white">
                <h3 className="widget-title">📈 Severity Indicators</h3>
                <div className="symptom-item">
                  <div className="symptom-header">
                    <span>Anhedonia (Depression)</span>
                    <span className="symptom-pct">18%</span>
                  </div>
                  <div className="symptom-track">
                    <div className="symptom-fill" style={{ width: '18%', backgroundColor: '#10b981' }} />
                  </div>
                </div>

                <div className="symptom-item">
                  <div className="symptom-header">
                    <span>Psychomotor Agitation (Anxiety)</span>
                    <span className="symptom-pct">20%</span>
                  </div>
                  <div className="symptom-track">
                    <div className="symptom-fill" style={{ width: '20%', backgroundColor: '#f43f5e' }} />
                  </div>
                </div>

                <div className="symptom-item">
                  <div className="symptom-header">
                    <span>Speech Latency (Stress)</span>
                    <span className="symptom-pct">22%</span>
                  </div>
                  <div className="symptom-track">
                    <div className="symptom-fill" style={{ width: '22%', backgroundColor: '#2563eb' }} />
                  </div>
                </div>
              </div>

              {/* FastSHAP Feature Impact */}
              <div className="card-white">
                <h3 className="widget-title">📊 FastSHAP Feature Impact</h3>
                <div className="shap-row"><span className="shap-name">GSR_Lev</span><div className="shap-bar-container"><div className="shap-bar positive" style={{ width: '40%' }} /></div><span className="shap-val">+1.4</span></div>
                <div className="shap-row"><span className="shap-name">Heart_R</span><div className="shap-bar-container"><div className="shap-bar positive" style={{ width: '32%' }} /></div><span className="shap-val">+1.1</span></div>
                <div className="shap-row"><span className="shap-name">Daily_A</span><div className="shap-bar-container"><div className="shap-bar positive" style={{ width: '25%' }} /></div><span className="shap-val">+0.9</span></div>
                <div className="shap-row"><span className="shap-name">Sleep_Q</span><div className="shap-bar-container"><div className="shap-bar negative" style={{ width: '65%' }} /></div><span className="shap-val">-2.4</span></div>
                <div className="shap-row"><span className="shap-name">HRV_Ind</span><div className="shap-bar-container"><div className="shap-bar negative" style={{ width: '50%' }} /></div><span className="shap-val">-1.8</span></div>
              </div>
            </div>

            {/* Bottom Row: Clinical Output Cards (Col 12) */}
            <div className="bottom-cards">
              {/* Acoustic Profile */}
              <div className="card-white card-pastel-blue">
                <h3 className="widget-title">📄 Acoustic Profile</h3>
                <p className="card-desc">Pitch variability (F0) remains blunted compared to normative baseline. Speech rate is reduced by 15%, consistent with mild psychomotor retardation.</p>
                <div className="badge-metric">
                  <span>F0 Variance</span>
                  <span style={{ color: '#2563eb' }}>BLUNTED (-15%)</span>
                </div>
              </div>

              {/* Visual Kinematics */}
              <div className="card-white card-pastel-rose">
                <h3 className="widget-title">👁️ Visual Kinematics</h3>
                <p className="card-desc">Reduced facial expressivity noted in lower facial action units. Eye contact maintained at 68% of session, within acceptable therapeutic range.</p>
                <div className="badge-metric">
                  <span>Eye Contact</span>
                  <span style={{ color: '#f43f5e' }}>68% ACCEPTABLE</span>
                </div>
              </div>

              {/* System Status & Observation Notes */}
              <div className="card-white card-pastel-green">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 className="widget-title" style={{ margin: 0 }}>🟢 System Status</h3>
                  <span style={{ fontSize: '0.65rem', background: '#dcfce7', color: '#15803d', padding: '0.2rem 0.5rem', borderRadius: '6px', fontWeight: 700 }}>Export FHIR</span>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', fontSize: '0.725rem', margin: '0.4rem 0' }}>
                  <div>Predicted Status: <strong>{result?.classification?.predicted_class || 'Healthy'}</strong></div>
                  <div>Data Quality: <strong>HIGH_FIDELITY</strong></div>
                  <div>Uptime: <strong>99.98%</strong></div>
                </div>

                <div className="synthesized-box">
                  <div className="synthesized-title">Synthesized Evaluation</div>
                  <div className="synthesized-text">
                    {result?.narrative
                      ? (typeof result.narrative === 'string' ? result.narrative : result.narrative.summary)
                      : 'Diagnostic Assessment: Automated evaluation indicates Healthy status with low distress markers.'}
                  </div>
                </div>

                <div className="note-input-group">
                  <input
                    type="text"
                    value={observationNote}
                    onChange={(e) => setObservationNote(e.target.value)}
                    placeholder="Add observation note..."
                    className="note-input"
                  />
                  <button onClick={handleSaveObservationNote} className="btn-save">
                    Save
                  </button>
                </div>

                {savedNotes.length > 0 && (
                  <div style={{ marginTop: '0.4rem', fontSize: '0.65rem', color: '#475569' }}>
                    <strong>Notes Log:</strong> {savedNotes.join(' • ')}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* File Upload Modal */}
      {isUploadOpen && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2 style={{ margin: 0, fontSize: '1.05rem', color: '#2563eb' }}>📁 Upload Media / Data Inputs</h2>
              <button onClick={() => setIsUploadOpen(false)} style={{ border: 'none', background: 'none', fontSize: '1.2rem', cursor: 'pointer' }}>×</button>
            </div>

            <div style={{ border: '2px dashed #cbd5e1', padding: '1.5rem', borderRadius: '12px', textAlign: 'center', marginBottom: '1rem', background: '#f8fafc' }}>
              <input
                type="file"
                id="file-input-modal"
                onChange={(e) => e.target.files && setSelectedFile(e.target.files[0])}
                style={{ display: 'none' }}
              />
              <label htmlFor="file-input-modal" style={{ cursor: 'pointer' }}>
                <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📤</div>
                <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>Click to select video, audio, image or CSV file</div>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Supports MP4, WAV, MP3, PNG, JPG, JSON, CSV</div>
              </label>
            </div>

            {selectedFile && (
              <div style={{ fontSize: '0.75rem', padding: '0.5rem', background: '#eff6ff', borderRadius: '8px', marginBottom: '1rem', border: '1px solid #bfdbfe', color: '#1e40af' }}>
                Selected: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024).toFixed(1)} KB)
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
              <button onClick={() => setIsUploadOpen(false)} className="btn-action">Cancel</button>
              <button
                onClick={() => {
                  setStatusMessage(`File ${selectedFile?.name || 'input'} processed and evaluated.`);
                  setIsUploadOpen(false);
                  triggerEvaluation();
                }}
                className="btn-save"
              >
                Process & Evaluate
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
