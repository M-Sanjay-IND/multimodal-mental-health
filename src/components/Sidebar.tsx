'use client';

import React, { memo, useState } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

export interface SidebarProps {
  onSessionToggle?: () => void;
  isSessionActive?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = memo(function Sidebar({
  onSessionToggle,
  isSessionActive = false,
}) {
  const {
    connectionState,
    connectWebSocket,
    disconnectWebSocket,
    isPaused,
    togglePause,
    lowPowerMode,
    toggleLowPowerMode,
    startBaselineCalibration,
    baselineCalibrated,
    simulateModalityDisruption,
    modalityStatus,
    samplingRateHz,
  } = useDiagnosticResults();

  const [showMetadataDrawer, setShowMetadataDrawer] = useState(false);

  return (
    <aside className="bg-[#121216] border-r border-[#1E1E24] w-64 flex flex-col justify-between p-4 flex-shrink-0 font-sans text-xs">
      <div className="space-y-4">
        {/* Pre-Assessment Baseline Calibration Button */}
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 font-mono mb-2">
            Patient Calibration
          </h2>
          <button
            onClick={startBaselineCalibration}
            className={`w-full py-2 px-3 rounded font-mono text-xs font-semibold tracking-wide transition-all cursor-pointer flex items-center justify-between border ${
              baselineCalibrated
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/20'
            }`}
          >
            <span>{baselineCalibrated ? 'Baseline Calibrated ✓' : 'Start 15s Calibration'}</span>
            <span className={`w-2 h-2 rounded-full ${baselineCalibrated ? 'bg-emerald-400' : 'bg-emerald-400 animate-pulse'}`} />
          </button>
          <span className="text-[10px] text-zinc-500 font-mono mt-1 block">
            {baselineCalibrated ? 'Personalized baseline active' : 'Normalizes EAR, F0 & GSR relative to patient baseline'}
          </span>
        </div>

        {/* Clinician Artifact Control: Pause Epoch Button */}
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 font-mono mb-2">
            Artifact Control
          </h2>
          <button
            onClick={togglePause}
            className={`w-full py-2 px-3 rounded font-mono text-xs font-semibold tracking-wide transition-all cursor-pointer flex items-center justify-between border ${
              isPaused
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 hover:bg-amber-500/30'
                : 'bg-[#0A0A0C] text-zinc-300 border-[#1E1E24] hover:bg-[#1E1E24]'
            }`}
          >
            <span>{isPaused ? 'Resume Session' : 'Pause / Invalidate Epoch'}</span>
            <span className={`w-2 h-2 rounded-full ${isPaused ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400'}`} />
          </button>
          <span className="text-[10px] text-zinc-500 font-mono mt-1 block">
            {isPaused ? 'Epoch paused (pruned last 3s spikes)' : 'Click during patient movement to prune last 3s spikes'}
          </span>
        </div>

        {/* Live Demo Trigger: Simulated Modality Disruption */}
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 font-mono mb-2">
            Live Demo Controls
          </h2>
          <button
            onClick={() => simulateModalityDisruption('visual')}
            className={`w-full py-2 px-3 rounded font-mono text-xs font-semibold tracking-wide transition-all cursor-pointer flex items-center justify-between border ${
              modalityStatus.visual !== 'active'
                ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                : 'bg-[#0A0A0C] text-zinc-300 border-[#1E1E24] hover:bg-[#1E1E24]'
            }`}
          >
            <span>{modalityStatus.visual !== 'active' ? 'Restore Visual Stream' : 'Simulate Camera Drop'}</span>
            <span className={`w-2 h-2 rounded-full ${modalityStatus.visual !== 'active' ? 'bg-rose-400 animate-ping' : 'bg-zinc-600'}`} />
          </button>
          <span className="text-[10px] text-zinc-500 font-mono mt-1 block">
            {modalityStatus.visual !== 'active' ? 'Camera stream degraded (confidence margin expanded)' : 'Simulates occlusion to demonstrate MARG confidence expansion'}
          </span>
        </div>

        {/* Low Power Mode & Adaptive Sampling Rate Readout */}
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 font-mono mb-2 flex items-center justify-between">
            <span>Terminal Hardware</span>
            <span className="text-[10px] text-cyan-400 font-normal">{samplingRateHz} Hz Sampling</span>
          </h2>
          <button
            onClick={toggleLowPowerMode}
            className={`w-full py-2 px-3 rounded font-mono text-xs font-semibold tracking-wide transition-all cursor-pointer flex items-center justify-between border ${
              lowPowerMode
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 hover:bg-cyan-500/30'
                : 'bg-[#0A0A0C] text-zinc-300 border-[#1E1E24] hover:bg-[#1E1E24]'
            }`}
          >
            <span>Terminal Low-Power Mode</span>
            <span className={`w-2 h-2 rounded-full ${lowPowerMode ? 'bg-cyan-400' : 'bg-zinc-600'}`} />
          </button>
        </div>

        {/* Expandable Technical Metadata Drawer */}
        <div>
          <button
            onClick={() => setShowMetadataDrawer((prev) => !prev)}
            className="w-full text-left font-mono text-[11px] text-zinc-400 hover:text-zinc-200 flex items-center justify-between py-1 border-b border-[#1E1E24] cursor-pointer"
          >
            <span>Technical Protocol Specs</span>
            <span>{showMetadataDrawer ? '▲' : '▼'}</span>
          </button>
          {showMetadataDrawer && (
            <div className="p-3 rounded bg-[#0A0A0C] border border-[#1E1E24] text-[10px] text-zinc-400 leading-relaxed font-mono mt-2 space-y-1">
              <div className="text-emerald-400 font-semibold">Zero-Retention Vectorization</div>
              <div>• Protocol: 1,632B ArrayBuffer</div>
              <div>• Sampling: {samplingRateHz} Hz Adaptive Bitrate</div>
              <div>• Network Latency Target: ≤25ms</div>
            </div>
          )}
        </div>
      </div>

      {/* Network Quick Control */}
      <div className="pt-3 border-t border-[#1E1E24] font-mono text-[11px]">
        <button
          onClick={() => {
            if (connectionState === 'CONNECTED') {
              disconnectWebSocket();
            } else {
              connectWebSocket();
            }
          }}
          className="w-full py-2 rounded border border-[#1E1E24] bg-[#0A0A0C] text-zinc-300 hover:bg-[#1E1E24] transition-colors cursor-pointer"
        >
          {connectionState === 'CONNECTED' ? 'Disconnect Socket' : 'Reconnect Socket'}
        </button>
      </div>
    </aside>
  );
});
