'use client';

import React, { memo, useEffect, useRef, useState, useCallback } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

interface FacialSaliencyOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isStreaming: boolean;
}

export const FacialSaliencyOverlay: React.FC<FacialSaliencyOverlayProps> = memo(
  function FacialSaliencyOverlay({ videoRef, isStreaming }) {
    const { saliencyWeights, continuousScores } = useDiagnosticResults();
    const salCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const [snapshotUrl, setSnapshotUrl] = useState<string | null>(null);
    const [audioBlobUrl, setAudioBlobUrl] = useState<string | null>(null);
    const [isRecordingAudio, setIsRecordingAudio] = useState<boolean>(false);
    const [recordDurationSec, setRecordDurationSec] = useState<number>(0);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<number | null>(null);

    // Dynamic Monochrome Affective Metrics
    const valenceVal = (-0.42 + (continuousScores.depression / 34) * -0.2).toFixed(2);
    const arousalVal = (0.68 + (continuousScores.stress / 39) * 0.15).toFixed(2);

    useEffect(() => {
      const canvas = waveCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      let phase = 0;
      let animId: number;

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

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();

        phase += 0.08;
        animId = requestAnimationFrame(drawWave);
      };

      animId = requestAnimationFrame(drawWave);

      return () => {
        cancelAnimationFrame(animId);
      };
    }, [isStreaming]);

    const handleCaptureSnapshot = useCallback(() => {
      if (videoRef.current) {
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth || 640;
        canvas.height = videoRef.current.videoHeight || 360;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          setSnapshotUrl(canvas.toDataURL('image/png'));
        }
      }
    }, [videoRef]);

    const handleToggleRecordAudio = useCallback(async () => {
      if (!isRecordingAudio) {
        try {
          const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const recorder = new MediaRecorder(audioStream);
          mediaRecorderRef.current = recorder;
          audioChunksRef.current = [];

          recorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunksRef.current.push(e.data);
          };

          recorder.onstop = () => {
            const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
            setAudioBlobUrl(URL.createObjectURL(blob));
          };

          recorder.start();
          setIsRecordingAudio(true);
          setRecordDurationSec(0);

          if (timerRef.current) clearInterval(timerRef.current);
          timerRef.current = window.setInterval(() => {
            setRecordDurationSec((prev) => prev + 1);
          }, 1000);
        } catch (err) {
          console.warn('Microphone error:', err);
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
      <div className="mono-card p-4 flex flex-col h-full space-y-4 font-sans text-white">
        {/* Video Viewport Container */}
        <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-black border border-zinc-800 flex-1">
          <video
            ref={videoRef}
            className={`w-full h-full object-cover transform -scale-x-100 ${
              isStreaming ? 'block grayscale brightness-90 contrast-125' : 'hidden'
            }`}
            playsInline
            muted
          />

          {!isStreaming && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950 text-zinc-500 font-mono text-xs space-y-2">
              <div className="w-10 h-10 rounded-full border border-zinc-800 flex items-center justify-center text-zinc-400 font-bold">
                CAM
              </div>
              <span>CAMERA STREAM STANDBY</span>
            </div>
          )}

          {/* Saliency Heatmap Canvas */}
          <canvas
            ref={salCanvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none transform -scale-x-100 z-10"
          />

          {/* Minimalist Status Badge */}
          <div className="absolute top-3 right-3 px-3 py-1 bg-black/80 backdrop-blur-md rounded-full border border-zinc-700 text-zinc-300 font-mono text-[11px] flex items-center gap-2 z-20">
            <span
              className={`w-2 h-2 rounded-full ${
                isStreaming ? 'bg-white animate-pulse' : 'bg-zinc-600'
              }`}
            />
            {isStreaming ? 'LIVE CAMERA ACTIVE' : 'STREAM STANDBY'}
          </div>

          {/* Minimal Affective Metrics Overlay */}
          <div className="absolute bottom-3 left-3 bg-black/80 backdrop-blur-md rounded-lg p-3 border border-zinc-800 font-mono text-[11px] space-y-1.5 z-20 text-zinc-300">
            <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-bold">Valence & Arousal</div>
            <div className="flex gap-4">
              <span>VAL: <strong className="text-white">{valenceVal}</strong></span>
              <span>ARO: <strong className="text-white">{arousalVal}</strong></span>
            </div>
          </div>

          {/* Bottom Waveform Line */}
          <div className="absolute bottom-0 left-0 w-full h-4 z-10 pointer-events-none opacity-50">
            <canvas ref={waveCanvasRef} className="w-full h-full" />
          </div>
        </div>

        {/* Action Controls Bar */}
        <div className="flex items-center justify-between gap-3 font-mono text-xs">
          <button
            onClick={handleCaptureSnapshot}
            className="flex-1 py-2 px-3 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-200 font-semibold transition-all cursor-pointer text-center"
          >
            Capture Frame
          </button>

          <button
            onClick={handleToggleRecordAudio}
            className={`flex-1 py-2 px-3 rounded-lg border font-semibold transition-all cursor-pointer text-center ${
              isRecordingAudio
                ? 'bg-white text-black border-white animate-pulse'
                : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-700 text-zinc-200'
            }`}
          >
            {isRecordingAudio ? `Stop Recording (${recordDurationSec}s)` : 'Record Mic Audio'}
          </button>
        </div>

        {/* Captured Preview Panel */}
        {(snapshotUrl || audioBlobUrl) && (
          <div className="p-3 bg-zinc-900/60 rounded-lg border border-zinc-800 flex flex-wrap gap-4 items-center justify-between font-mono text-xs">
            <div className="flex items-center gap-4">
              {snapshotUrl && (
                <div className="flex items-center gap-2">
                  <img src={snapshotUrl} alt="Captured frame" className="w-12 h-8 object-cover rounded border border-zinc-700" />
                  <span className="text-[11px] text-zinc-400">Frame Captured</span>
                </div>
              )}

              {audioBlobUrl && (
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-zinc-400">Audio Recorded:</span>
                  <audio src={audioBlobUrl} controls className="h-6 max-w-[160px]" />
                </div>
              )}
            </div>

            <button
              onClick={() => {
                setSnapshotUrl(null);
                setAudioBlobUrl(null);
              }}
              className="text-[10px] text-zinc-500 hover:text-zinc-300 underline cursor-pointer"
            >
              Clear Media
            </button>
          </div>
        )}

        {/* Facial Action Units Breakdown */}
        <div className="grid grid-cols-4 gap-2 font-mono text-[11px] text-center">
          <div className="bg-zinc-900 p-2 rounded border border-zinc-800">
            <span className="text-[9px] text-zinc-500 block">AU04 Brow</span>
            <span className="font-bold text-zinc-200">{(saliencyWeights.au04BrowLowerer || 0.15).toFixed(2)}</span>
          </div>
          <div className="bg-zinc-900 p-2 rounded border border-zinc-800">
            <span className="text-[9px] text-zinc-500 block">AU15 Lip</span>
            <span className="font-bold text-zinc-200">{(saliencyWeights.au15LipDepressor || 0.12).toFixed(2)}</span>
          </div>
          <div className="bg-zinc-900 p-2 rounded border border-zinc-800">
            <span className="text-[9px] text-zinc-500 block">AU06 Cheek</span>
            <span className="font-bold text-zinc-200">{(saliencyWeights.au06CheekRaiser || 0.65).toFixed(2)}</span>
          </div>
          <div className="bg-zinc-900 p-2 rounded border border-zinc-800">
            <span className="text-[9px] text-zinc-500 block">AU12 Smile</span>
            <span className="font-bold text-zinc-200">{(saliencyWeights.au12SmilePuller || 0.58).toFixed(2)}</span>
          </div>
        </div>
      </div>
    );
  }
);
