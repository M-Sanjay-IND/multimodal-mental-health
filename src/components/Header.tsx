'use client';

import React, { memo } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

export interface HeaderProps {
  onSessionToggle?: () => void;
  isSessionActive?: boolean;
}

export const Header: React.FC<HeaderProps> = memo(function Header({
  onSessionToggle,
  isSessionActive = false,
}) {
  const { connectionState, modalityStatus, payloadLatencyMs } = useDiagnosticResults();

  const getStatusBadge = (status: 'active' | 'degraded' | 'disabled') => {
    switch (status) {
      case 'active':
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
      case 'degraded':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      case 'disabled':
      default:
        return 'bg-zinc-800 text-zinc-500 border-zinc-700';
    }
  };

  return (
    <header className="bg-[#121216] border-b border-[#1E1E24] px-6 h-14 flex items-center justify-between z-40 flex-shrink-0">
      {/* Title & Brand */}
      <div className="flex items-center gap-3">
        <div className="w-2.5 h-2.5 rounded-full bg-[#00FF66]" />
        <div>
          <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">
            Multimodal Psychiatric Evaluation Engine
          </h1>
          <p className="text-[11px] font-mono text-zinc-400">
            Real-Time Privacy-Preserving Clinical Telemetry
          </p>
        </div>
      </div>

      {/* Stream Modalities & Connection Status */}
      <div className="flex items-center gap-4 text-xs font-mono">
        {/* Modality Badges */}
        <div className="hidden md:flex items-center gap-2 border-r border-[#1E1E24] pr-4">
          <span className={`px-2 py-0.5 rounded border text-[10px] uppercase ${getStatusBadge(modalityStatus.visual)}`}>
            Visual: {modalityStatus.visual}
          </span>
          <span className={`px-2 py-0.5 rounded border text-[10px] uppercase ${getStatusBadge(modalityStatus.acoustic)}`}>
            Acoustic: {modalityStatus.acoustic}
          </span>
          <span className={`px-2 py-0.5 rounded border text-[10px] uppercase ${getStatusBadge(modalityStatus.tabular)}`}>
            Tabular: {modalityStatus.tabular}
          </span>
        </div>

        {/* Latency Readout */}
        <div className="flex items-center gap-1.5 text-zinc-400">
          <span>Latency:</span>
          <span className="text-[#00FF66] font-semibold">{payloadLatencyMs} ms</span>
        </div>

        {/* Connection State */}
        <div className="flex items-center gap-2 bg-[#0A0A0C] border border-[#1E1E24] px-3 py-1 rounded">
          <span
            className={`w-2 h-2 rounded-full ${
              connectionState === 'CONNECTED'
                ? 'bg-[#00FF66]'
                : connectionState === 'CONNECTING' || connectionState === 'RECONNECTING'
                ? 'bg-[#FFB800] animate-pulse'
                : 'bg-zinc-600'
            }`}
          />
          <span className="text-zinc-300 font-medium text-[11px]">{connectionState}</span>
        </div>

        {/* Session Action Button */}
        {onSessionToggle && (
          <button
            onClick={onSessionToggle}
            className={`px-4 py-1.5 rounded text-xs font-semibold font-mono tracking-wide transition-all cursor-pointer ${
              isSessionActive
                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30'
                : 'bg-[#00FF66] text-black hover:bg-emerald-400 font-bold'
            }`}
          >
            {isSessionActive ? 'Stop Session' : 'Start Session'}
          </button>
        )}
      </div>
    </header>
  );
});
