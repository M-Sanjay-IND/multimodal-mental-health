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
  const [activeTab, setActiveTab] = useState('Dashboard');

  return (
    <nav className="hidden md:flex flex-col py-internal-padding gap-stack-sm bg-surface-container-lowest text-clinical-blue font-body-md text-body-md fixed left-0 top-0 h-full w-nav-width shadow-sm border-r border-outline-variant z-40 overflow-y-auto">
      {/* Clinician Profile Avatar */}
      <div className="px-container-padding mb-4 flex flex-col items-center">
        <div className="w-16 h-16 rounded-full bg-surface-container-low mb-3 overflow-hidden shadow-sm border border-outline-variant">
          <img
            className="w-full h-full object-cover"
            alt="Dr. Adrian Sterling profile"
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuDRv7L6bUPCmdf7wgNHHq4MEWpakF3Rkec6gPZYQLG-JuYyOggBN_-LJ0eCsdgl87ozPgKEz8cXp1w7uVnwZ6n2ni7U0p1yhaoLUemkrsQWGuL5aIs_mtL6XHQ21_bthZXaXu1d9iJb-BAdNARvXEYWKDnZZOptZUN7q_tMz6UXa--agSv2qwLKRpdH_9bknI0nNdi25T5W9W_WJDgP4ud8-ldJJKy-RdVrxzz1StLjCnZmhPakEl3H"
          />
        </div>
        <div className="font-nav-brand text-nav-brand text-clinical-blue text-center">
          Dr. Adrian Sterling
        </div>
        <div className="font-caption text-caption text-on-surface-variant mt-1 uppercase tracking-wider">
          Senior Psychiatrist
        </div>
      </div>

      {/* Primary Navigation Links */}
      <div className="px-4 space-y-1 mb-4">
        {[
          { label: 'Dashboard', icon: 'dashboard' },
          { label: 'Patient Monitoring', icon: 'videocam' },
          { label: 'Session Analysis', icon: 'psychology' },
          { label: 'Telemetry Data', icon: 'insights' },
          { label: 'Reports', icon: 'description' },
        ].map((item) => (
          <button
            key={item.label}
            onClick={() => setActiveTab(item.label)}
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-200 ease-in-out cursor-pointer font-semibold text-xs ${
              activeTab === item.label
                ? 'bg-secondary-fixed text-on-secondary-fixed-variant shadow-sm'
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
            }`}
          >
            <span
              className={`material-symbols-outlined ${
                activeTab === item.label ? 'text-clinical-blue' : ''
              }`}
            >
              {item.icon}
            </span>
            {item.label}
          </button>
        ))}
      </div>

      {/* Clinical Telemetry & Hardware Control Panel */}
      <div className="px-4 space-y-3 pt-3 border-t border-outline-variant text-xs">
        <div className="font-caption text-caption text-on-surface-variant uppercase tracking-wider font-bold">
          Clinical Tools
        </div>

        {/* 15s Baseline Calibration */}
        <button
          onClick={startBaselineCalibration}
          className={`w-full py-2 px-3 rounded-lg font-data-label text-xs font-semibold transition-all cursor-pointer flex items-center justify-between border ${
            baselineCalibrated
              ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
              : 'bg-surface-container-low text-on-surface border-outline-variant hover:bg-surface-container'
          }`}
        >
          <span className="flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[16px] text-clinical-blue">tune</span>
            {baselineCalibrated ? 'Baseline Active ✓' : '15s Calibration'}
          </span>
          <span
            className={`w-2 h-2 rounded-full ${
              baselineCalibrated ? 'bg-success-green' : 'bg-warning-amber animate-pulse'
            }`}
          />
        </button>

        {/* Epoch Invalidate / Pause */}
        <button
          onClick={togglePause}
          className={`w-full py-2 px-3 rounded-lg font-data-label text-xs font-semibold transition-all cursor-pointer flex items-center justify-between border ${
            isPaused
              ? 'bg-amber-50 text-amber-800 border-amber-300'
              : 'bg-surface-container-low text-on-surface border-outline-variant hover:bg-surface-container'
          }`}
        >
          <span className="flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[16px]">pause_circle</span>
            {isPaused ? 'Resume Session' : 'Pause Epoch'}
          </span>
          <span
            className={`w-2 h-2 rounded-full ${
              isPaused ? 'bg-warning-amber animate-pulse' : 'bg-success-green'
            }`}
          />
        </button>

        {/* Live Camera Occlusion Test */}
        <button
          onClick={() => simulateModalityDisruption('visual')}
          className={`w-full py-2 px-3 rounded-lg font-data-label text-xs font-semibold transition-all cursor-pointer flex items-center justify-between border ${
            modalityStatus.visual !== 'active'
              ? 'bg-rose-50 text-rose-700 border-rose-300'
              : 'bg-surface-container-low text-on-surface border-outline-variant hover:bg-surface-container'
          }`}
        >
          <span className="flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[16px]">no_photography</span>
            {modalityStatus.visual !== 'active' ? 'Restore Camera' : 'Simulate Occlusion'}
          </span>
        </button>

        {/* Hardware Low Power Mode */}
        <button
          onClick={toggleLowPowerMode}
          className={`w-full py-2 px-3 rounded-lg font-data-label text-xs font-semibold transition-all cursor-pointer flex items-center justify-between border ${
            lowPowerMode
              ? 'bg-blue-50 text-blue-800 border-blue-300'
              : 'bg-surface-container-low text-on-surface border-outline-variant hover:bg-surface-container'
          }`}
        >
          <span className="flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[16px]">battery_saver</span>
            Low Power ({samplingRateHz}Hz)
          </span>
          <span className={`w-2 h-2 rounded-full ${lowPowerMode ? 'bg-clinical-blue' : 'bg-outline'}`} />
        </button>

        {/* Technical Specs Accordion */}
        <div>
          <button
            onClick={() => setShowMetadataDrawer((prev) => !prev)}
            className="w-full text-left font-data-mono text-[11px] text-on-surface-variant hover:text-on-surface flex items-center justify-between py-1 cursor-pointer"
          >
            <span>Protocol Specs</span>
            <span className="material-symbols-outlined text-[16px]">
              {showMetadataDrawer ? 'expand_less' : 'expand_more'}
            </span>
          </button>
          {showMetadataDrawer && (
            <div className="p-2.5 rounded-lg bg-surface-container-low border border-outline-variant font-data-mono text-[10px] text-on-surface-variant space-y-1">
              <div className="text-clinical-blue font-bold">Zero-Retention Vector</div>
              <div>• Payload: 1,632B ArrayBuffer</div>
              <div>• Latency Target: ≤25ms</div>
              <div>• Status: {connectionState}</div>
            </div>
          )}
        </div>

        {/* Socket Reconnect */}
        <button
          onClick={() => {
            if (connectionState === 'CONNECTED') {
              disconnectWebSocket();
            } else {
              connectWebSocket();
            }
          }}
          className="w-full py-1.5 rounded-lg font-data-mono text-[11px] border border-outline-variant bg-surface-container-lowest hover:bg-surface-container text-on-surface transition-colors cursor-pointer"
        >
          {connectionState === 'CONNECTED' ? 'Disconnect Socket' : 'Reconnect Socket'}
        </button>
      </div>

      {/* Emergency Override & Footer */}
      <div className="px-4 mt-auto space-y-1 pt-4">
        <button
          onClick={() => alert('Emergency Override Triggered. Clinician notifications dispatched.')}
          className="w-full py-2.5 px-4 border border-alert-coral text-alert-coral rounded-xl font-bold flex items-center justify-center gap-2 hover:bg-alert-coral hover:text-white transition-colors cursor-pointer text-xs"
        >
          <span className="material-symbols-outlined text-[18px]">warning</span>
          EMERGENCY OVERRIDE
        </button>

        <div className="pt-3 border-t border-outline-variant space-y-0.5">
          <a
            className="flex items-center gap-3 px-4 py-2 text-on-surface-variant hover:text-on-surface rounded-xl hover:bg-surface-container transition-all text-xs"
            href="#"
          >
            <span className="material-symbols-outlined text-[18px]">contact_support</span>
            Support
          </a>
          <a
            className="flex items-center gap-3 px-4 py-2 text-on-surface-variant hover:text-on-surface rounded-xl hover:bg-surface-container transition-all text-xs"
            href="#"
          >
            <span className="material-symbols-outlined text-[18px]">logout</span>
            Sign Out
          </a>
        </div>
      </div>
    </nav>
  );
});

