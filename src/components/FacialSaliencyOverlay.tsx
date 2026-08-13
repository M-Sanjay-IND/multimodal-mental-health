'use client';

import React, { memo, useEffect, useRef } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

interface FacialSaliencyOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isStreaming: boolean;
}

export const FacialSaliencyOverlay: React.FC<FacialSaliencyOverlayProps> = memo(
  function FacialSaliencyOverlay({ videoRef, isStreaming }) {
    const { saliencyWeights, lowPowerMode } = useDiagnosticResults();
    const salCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const waveAnimFrameRef = useRef<number | null>(null);
    const salAnimFrameRef = useRef<number | null>(null);
    const lastDrawTimeRef = useRef<number>(0);

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

        ctx.strokeStyle = 'rgba(0, 255, 102, 0.6)';
        ctx.lineWidth = 1;
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
     * 2. Draw Landmark-Anchored Spatial Attention & Saliency Heatmaps
     * Respects lowPowerMode (throttling to 15 FPS when enabled on low-end terminals)
     */
    useEffect(() => {
      const canvas = salCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const drawSaliency = (timestamp: number) => {
        // Low Power Mode Frame Throttling (15 FPS = 66ms interval)
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
          const faceWidth = canvas.width * 0.38;
          const faceHeight = canvas.height * 0.52;

          // Spatial Action Unit Heatmaps derived from model saliencyWeights
          const au04Val = saliencyWeights.au04BrowLowerer || 0.15;
          const au04Radius = 12 + au04Val * 20;
          const grad04 = ctx.createRadialGradient(
            centerX,
            centerY - faceHeight * 0.25,
            2,
            centerX,
            centerY - faceHeight * 0.25,
            au04Radius
          );
          grad04.addColorStop(0, `rgba(255, 51, 102, ${Math.min(0.7, au04Val + 0.2)})`);
          grad04.addColorStop(1, 'rgba(255, 51, 102, 0)');
          ctx.fillStyle = grad04;
          ctx.beginPath();
          ctx.arc(centerX, centerY - faceHeight * 0.25, au04Radius, 0, Math.PI * 2);
          ctx.fill();

          const au15Val = saliencyWeights.au15LipDepressor || 0.12;
          const au15Radius = 10 + au15Val * 18;
          const grad15 = ctx.createRadialGradient(
            centerX,
            centerY + faceHeight * 0.28,
            2,
            centerX,
            centerY + faceHeight * 0.28,
            au15Radius
          );
          grad15.addColorStop(0, `rgba(255, 184, 0, ${Math.min(0.6, au15Val + 0.2)})`);
          grad15.addColorStop(1, 'rgba(255, 184, 0, 0)');
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

    return (
      <div className="bg-[#121216] border border-[#1E1E24] rounded p-4 flex flex-col justify-between h-full">
        {/* Header */}
        <div className="flex items-center justify-between mb-3 border-b border-[#1E1E24] pb-2 font-mono text-xs">
          <span className="font-semibold text-zinc-200">Spatial Attention &amp; Video Stream</span>
          <div className="flex items-center gap-2">
            {lowPowerMode && (
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                15 FPS LOW POWER
              </span>
            )}
            <span
              className={`px-2 py-0.5 rounded border text-[10px] uppercase font-semibold ${
                isStreaming
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  : 'bg-zinc-800 text-zinc-500 border-zinc-700'
              }`}
            >
              {isStreaming ? 'Stream Active' : 'Standby'}
            </span>
          </div>
        </div>

        {/* Video Viewport Container */}
        <div className="relative w-full aspect-video bg-[#0A0A0C] rounded overflow-hidden border border-[#1E1E24] flex items-center justify-center min-h-[260px]">
          <video
            ref={videoRef}
            className="w-full h-full object-cover opacity-80 transform -scale-x-100"
            playsInline
            muted
          />

          {/* Saliency Overlay Canvas */}
          <canvas
            ref={salCanvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none transform -scale-x-100"
          />

          {!isStreaming && (
            <div className="absolute inset-0 bg-[#0A0A0C]/90 flex flex-col items-center justify-center gap-1 font-mono text-xs text-zinc-500">
              <span>Camera Stream Inactive</span>
              <span className="text-[10px] text-zinc-600">Click "Start Session" to initiate stream</span>
            </div>
          )}

          {/* Bottom Audio Waveform Bar */}
          <div className="absolute bottom-0 left-0 w-full h-6 bg-[#0A0A0C]/60 border-t border-[#1E1E24] backdrop-blur-sm overflow-hidden flex items-end">
            <canvas ref={waveCanvasRef} className="w-full h-full" />
          </div>
        </div>

        {/* Action Unit Metrics Bar */}
        <div className="mt-3 grid grid-cols-4 gap-2 font-mono text-[11px]">
          <div className="p-2 rounded bg-[#0A0A0C] border border-[#1E1E24] text-center">
            <span className="text-zinc-500 text-[10px] block">AU04 Brow Lower</span>
            <span className="text-zinc-200 font-semibold">
              {(saliencyWeights.au04BrowLowerer || 0.15).toFixed(2)}
            </span>
          </div>
          <div className="p-2 rounded bg-[#0A0A0C] border border-[#1E1E24] text-center">
            <span className="text-zinc-500 text-[10px] block">AU15 Lip Depress</span>
            <span className="text-zinc-200 font-semibold">
              {(saliencyWeights.au15LipDepressor || 0.12).toFixed(2)}
            </span>
          </div>
          <div className="p-2 rounded bg-[#0A0A0C] border border-[#1E1E24] text-center">
            <span className="text-zinc-500 text-[10px] block">AU06 Cheek Raise</span>
            <span className="text-[#00FF66] font-semibold">
              {(saliencyWeights.au06CheekRaiser || 0.65).toFixed(2)}
            </span>
          </div>
          <div className="p-2 rounded bg-[#0A0A0C] border border-[#1E1E24] text-center">
            <span className="text-zinc-500 text-[10px] block">AU12 Smile Pull</span>
            <span className="text-[#00FF66] font-semibold">
              {(saliencyWeights.au12SmilePuller || 0.58).toFixed(2)}
            </span>
          </div>
        </div>
      </div>
    );
  }
);
