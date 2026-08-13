'use client';

import React, { memo, useEffect, useRef, useState, useCallback } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

interface FacialSaliencyOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isStreaming: boolean;
}

export const FacialSaliencyOverlay: React.FC<FacialSaliencyOverlayProps> = memo(
  function FacialSaliencyOverlay({ videoRef, isStreaming }) {
    const { saliencyWeights, lowPowerMode, continuousScores } = useDiagnosticResults();
    const salCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const waveAnimFrameRef = useRef<number | null>(null);
    const salAnimFrameRef = useRef<number | null>(null);
    const lastDrawTimeRef = useRef<number>(0);

    // Media capture states
    const [snapshotUrl, setSnapshotUrl] = useState<string | null>(null);
    const [audioBlobUrl, setAudioBlobUrl] = useState<string | null>(null);
    const [isRecordingAudio, setIsRecordingAudio] = useState<boolean>(false);
    const [recordDurationSec, setRecordDurationSec] = useState<number>(0);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<number | null>(null);

    // Compute dynamic Valence & Arousal values
    const valenceVal = (-0.42 + (continuousScores.depression / 34) * -0.2).toFixed(2);
    const valencePct = Math.min(100, Math.max(10, Math.round(Math.abs(parseFloat(valenceVal)) * 100)));

    const arousalVal = (0.68 + (continuousScores.stress / 39) * 0.15).toFixed(2);
    const arousalPct = Math.min(100, Math.max(10, Math.round(parseFloat(arousalVal) * 100)));

    /**
     * 1. Draw Audio Waveform Canvas at bottom of viewport
     */
    useEffect(() => {
      const canvas = waveCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      let phase = 0;

      const drawWave = () => {
        if (canvas.parentElement) {
          canvas.width = canvas.parentElement.clientWidth;
          canvas.height = canvas.parentElement.clientHeight;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);

        for (let x = 0; x < canvas.width; x++) {
          const amp = isStreaming ? 6 : 1.5;
          let y = Math.sin(x * 0.04 + phase) * (Math.random() * amp + 1);
          y += Math.sin(x * 0.08 + phase * 1.5) * (amp / 2);
          ctx.lineTo(x, canvas.height / 2 + y);
        }

        ctx.strokeStyle = 'rgba(37, 99, 235, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        phase += 0.08;
        waveAnimFrameRef.current = requestAnimationFrame(drawWave);
      };

      waveAnimFrameRef.current = requestAnimationFrame(drawWave);

      return () => {
        if (waveAnimFrameRef.current) cancelAnimationFrame(waveAnimFrameRef.current);
      };
    }, [isStreaming]);

    /**
     * 2. Draw Landmark Heatmaps
     */
    useEffect(() => {
      const canvas = salCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const drawSaliency = (timestamp: number) => {
        if (lowPowerMode && timestamp - lastDrawTimeRef.current < 66) {
          salAnimFrameRef.current = requestAnimationFrame(drawSaliency);
          return;
        }
        lastDrawTimeRef.current = timestamp;

        if (canvas.parentElement) {
          canvas.width = canvas.parentElement.clientWidth;
          canvas.height = canvas.parentElement.clientHeight;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (isStreaming) {
          const centerX = canvas.width / 2;
          const centerY = canvas.height / 2;
          const faceHeight = canvas.height * 0.52;

          const au04Val = saliencyWeights.au04BrowLowerer || 0.15;
          const au04Radius = 14 + au04Val * 22;
          const grad04 = ctx.createRadialGradient(
            centerX,
            centerY - faceHeight * 0.25,
            2,
            centerX,
            centerY - faceHeight * 0.25,
            au04Radius
          );
          grad04.addColorStop(0, `rgba(244, 63, 94, ${Math.min(0.7, au04Val + 0.2)})`);
          grad04.addColorStop(1, 'rgba(244, 63, 94, 0)');
          ctx.fillStyle = grad04;
          ctx.beginPath();
          ctx.arc(centerX, centerY - faceHeight * 0.25, au04Radius, 0, Math.PI * 2);
          ctx.fill();

          const au15Val = saliencyWeights.au15LipDepressor || 0.12;
          const au15Radius = 12 + au15Val * 20;
          const grad15 = ctx.createRadialGradient(
            centerX,
            centerY + faceHeight * 0.28,
            2,
            centerX,
            centerY + faceHeight * 0.28,
            au15Radius
          );
          grad15.addColorStop(0, `rgba(245, 158, 11, ${Math.min(0.6, au15Val + 0.2)})`);
          grad15.addColorStop(1, 'rgba(245, 158, 11, 0)');
          ctx.fillStyle = grad15;
          ctx.beginPath();
          ctx.arc(centerX, centerY + faceHeight * 0.28, au15Radius, 0, Math.PI * 2);
          ctx.fill();
        }

        salAnimFrameRef.current = requestAnimationFrame(drawSaliency);
      };

      salAnimFrameRef.current = requestAnimationFrame(drawSaliency);

      return () => {
        if (salAnimFrameRef.current) cancelAnimationFrame(salAnimFrameRef.current);
      };
    }, [isStreaming, lowPowerMode, saliencyWeights]);

    /**
     * 3. Capture Real Photo Snapshot from Video Feed
     */
    const handleCaptureSnapshot = useCallback(() => {
      if (videoRef.current) {
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth || 640;
        canvas.height = videoRef.current.videoHeight || 360;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL('image/png');
          setSnapshotUrl(dataUrl);
        }
      }
    }, [videoRef]);

    /**
     * 4. Record Real Microphone Audio Clip Session
     */
    const handleToggleRecordAudio = useCallback(async () => {
      if (!isRecordingAudio) {
        try {
          const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
          };

          recorder.start();
          setIsRecordingAudio(true);
          setRecordDurationSec(0);

          if (timerRef.current) clearInterval(timerRef.current);
          timerRef.current = window.setInterval(() => {
            setRecordDurationSec((prev) => prev + 1);
          }, 1000);
        } catch (err) {
          console.warn('Microphone permission error:', err);
        }
      } else {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
        }
        if (timerRef.current) clearInterval(timerRef.current);
        setIsRecordingAudio(false);
      }
    }, [isRecordingAudio]);

    return (
      <div className="bg-surface-container-lowest rounded-[24px] pastel-shadow p-3 relative overflow-hidden flex flex-col h-full border border-border-subtle/50 space-y-3">
        <div className="relative w-full aspect-video rounded-[20px] overflow-hidden bg-black group flex-1">
          {/* Real Webcam Stream or Fallback Image */}
          <div className="absolute inset-0 bg-secondary flex items-center justify-center">
            <video
              ref={videoRef}
              className={`w-full h-full object-cover transform -scale-x-100 ${
                isStreaming ? 'block' : 'hidden'
              }`}
              playsInline
              muted
            />

            {!isStreaming && (
              <img
                className="w-full h-full object-cover opacity-80 mix-blend-luminosity"
                alt="Clinical patient webcam placeholder"
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuAH4mWaao3aEGr3OWTSVSvHYuqq1YAjI0P1eEFnbKONTu4GwzeEqJuhMtZHJhSmwC41vbg35c8i_tzw4piEQC1VpJ65sh-NZ-2mC2rd7tR1lzwPTWEvy7woTlKGxEj1c5WZ8qr_ug2rgcQ3kd19Bc1x5Q9UpyDJ1hDCdJLGMWm0SE4HDy1erxd8JhCGzJx-CkzTnCzALdF6j2ZT37VLSABAHFsYf4TM6xnPvvr_CVSJcnkPse3yaFO8"
              />
            )}
          </div>

          {/* Saliency Heatmap Canvas */}
          <canvas
            ref={salCanvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none transform -scale-x-100 z-10"
          />

          {/* Affective Computing Overlay (Glassmorphism) */}
          <div className="absolute bottom-4 left-4 glass-panel rounded-xl p-4 w-64 shadow-lg border border-white/20 transition-all duration-300 opacity-95 group-hover:opacity-100 z-20">
            <div className="font-data-label text-data-label text-on-surface mb-2 font-semibold tracking-wide">
              AFFECTIVE STATE
            </div>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between font-caption text-caption text-on-surface-variant mb-1">
                  <span>VALENCE</span>
                  <span>{valenceVal}</span>
                </div>
                <div className="h-1.5 w-full bg-black/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-clinical-blue/70 rounded-full relative transition-all duration-500"
                    style={{ width: `${valencePct}%` }}
                  >
                    <div className="absolute right-0 top-1/2 -translate-y-1/2 w-4 h-4 bg-clinical-blue/20 rounded-full blur-[2px]" />
                  </div>
                </div>
              </div>

              <div>
                <div className="flex justify-between font-caption text-caption text-on-surface-variant mb-1">
                  <span>AROUSAL</span>
                  <span>{arousalVal}</span>
                </div>
                <div className="h-1.5 w-full bg-black/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-alert-coral/60 rounded-full transition-all duration-500"
                    style={{ width: `${arousalPct}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Recording & Stream Indicator */}
          <div className="absolute top-4 right-4 px-3 py-1 bg-black/40 backdrop-blur-md rounded-full flex items-center gap-2 border border-white/10 text-white font-data-mono text-data-mono z-20">
            <span
              className={`w-2 h-2 rounded-full ${
                isStreaming ? 'bg-alert-coral animate-pulse' : 'bg-amber-400'
              }`}
            />
            {isStreaming ? 'REC 42:15' : 'STREAM STANDBY'}
          </div>

          {/* Bottom Audio Waveform Overlay Bar */}
          <div className="absolute bottom-0 left-0 w-full h-5 bg-black/20 backdrop-blur-sm z-10 pointer-events-none">
            <canvas ref={waveCanvasRef} className="w-full h-full" />
          </div>
        </div>

        {/* Quick Action Controls Bar (Camera Snapshot & Mic Recording) */}
        <div className="px-2 py-2 flex items-center justify-between gap-3 bg-surface-container-low/90 rounded-xl border border-outline-variant/60">
          <button
            onClick={handleCaptureSnapshot}
            className="flex-1 py-2 px-3 rounded-lg bg-surface-container-lowest hover:bg-surface-container border border-outline-variant text-xs font-semibold text-on-surface flex items-center justify-center gap-1.5 transition-all cursor-pointer shadow-sm"
          >
            <span className="material-symbols-outlined text-[18px] text-clinical-blue">photo_camera</span>
            Capture Snapshot Photo
          </button>

          <button
            onClick={handleToggleRecordAudio}
            className={`flex-1 py-2 px-3 rounded-lg border text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer shadow-sm ${
              isRecordingAudio
                ? 'bg-rose-50 text-rose-700 border-rose-300 animate-pulse'
                : 'bg-surface-container-lowest hover:bg-surface-container border-outline-variant text-on-surface'
            }`}
          >
            <span className="material-symbols-outlined text-[18px] text-alert-coral">mic</span>
            {isRecordingAudio ? `Stop Recording (${recordDurationSec}s)` : 'Record Audio Clip'}
          </button>
        </div>

        {/* Captured Media Preview Panel (Shows actual photo snapshot & playable audio player) */}
        {(snapshotUrl || audioBlobUrl) && (
          <div className="p-3 bg-surface-container-low rounded-xl border border-outline-variant/60 flex flex-wrap gap-4 items-center justify-between">
            <div className="flex items-center gap-4 flex-wrap">
              {snapshotUrl && (
                <div className="flex items-center gap-2.5 bg-surface-container-lowest p-1.5 rounded-lg border border-outline-variant">
                  <img src={snapshotUrl} alt="Captured camera photo" className="w-16 h-10 object-cover rounded border" />
                  <div>
                    <span className="font-bold text-xs block text-on-surface">Snapshot Frame</span>
                    <span className="text-[10px] text-on-surface-variant">PNG • Visual Embedding Extracted</span>
                  </div>
                </div>
              )}

              {audioBlobUrl && (
                <div className="flex flex-col gap-1 bg-surface-container-lowest p-2 rounded-lg border border-outline-variant">
                  <span className="font-bold text-xs text-on-surface">Recorded Audio Clip</span>
                  <audio src={audioBlobUrl} controls className="h-8 max-w-[200px]" />
                </div>
              )}
            </div>

            <button
              onClick={() => {
                setSnapshotUrl(null);
                setAudioBlobUrl(null);
              }}
              className="text-[11px] font-semibold text-on-surface-variant hover:text-on-surface underline cursor-pointer"
            >
              Clear Captured Media
            </button>
          </div>
        )}

        {/* Action Unit Metrics Bar below video */}
        <div className="px-2 py-1.5 grid grid-cols-4 gap-2 font-data-mono text-data-mono text-on-surface-variant">
          <div className="text-center bg-surface-container-low/60 rounded-lg p-1">
            <span className="text-[10px] text-on-surface-variant block">AU04 Brow</span>
            <span className="font-bold text-on-surface">
              {(saliencyWeights.au04BrowLowerer || 0.15).toFixed(2)}
            </span>
          </div>
          <div className="text-center bg-surface-container-low/60 rounded-lg p-1">
            <span className="text-[10px] text-on-surface-variant block">AU15 Lip</span>
            <span className="font-bold text-on-surface">
              {(saliencyWeights.au15LipDepressor || 0.12).toFixed(2)}
            </span>
          </div>
          <div className="text-center bg-surface-container-low/60 rounded-lg p-1">
            <span className="text-[10px] text-on-surface-variant block">AU06 Cheek</span>
            <span className="font-bold text-clinical-blue">
              {(saliencyWeights.au06CheekRaiser || 0.65).toFixed(2)}
            </span>
          </div>
          <div className="text-center bg-surface-container-low/60 rounded-lg p-1">
            <span className="text-[10px] text-on-surface-variant block">AU12 Smile</span>
            <span className="font-bold text-clinical-blue">
              {(saliencyWeights.au12SmilePuller || 0.58).toFixed(2)}
            </span>
          </div>
        </div>
      </div>
    );
  }
);
