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
  const { modalityStatus, payloadLatencyMs } = useDiagnosticResults();

  const getVisualPillStyle = () => {
    switch (modalityStatus.visual) {
      case 'active':
        return 'pastel-green';
      case 'degraded':
        return 'pastel-apricot';
      default:
        return 'bg-surface-container-high text-on-surface-variant';
    }
  };

  const getAcousticPillStyle = () => {
    switch (modalityStatus.acoustic) {
      case 'active':
        return 'pastel-green';
      case 'degraded':
        return 'pastel-apricot';
      default:
        return 'bg-surface-container-high text-on-surface-variant';
    }
  };

  return (
    <header className="fixed top-0 right-0 w-full z-50 h-top-bar-height flex items-center justify-between px-container-padding bg-surface text-clinical-blue font-nav-brand text-nav-brand shadow-sm border-b border-outline-variant md:w-[calc(100%-16rem)]">
      <div className="flex items-center gap-4">
        <span className="font-nav-brand text-nav-brand text-clinical-blue uppercase tracking-wider font-semibold">
          PSYCH-METRIC
        </span>
      </div>

      <div className="flex items-center gap-4 lg:gap-6">
        {/* Dynamic Telemetry Pills (Soft Pastel) */}
        <div className="hidden sm:flex items-center gap-2.5 font-data-mono text-data-mono">
          <span className={`px-3 py-1 rounded-full ${getVisualPillStyle()} flex items-center gap-1.5 shadow-sm text-xs font-medium`}>
            <span
              className={`w-2 h-2 rounded-full ${
                modalityStatus.visual === 'active'
                  ? 'bg-success-green animate-pulse'
                  : modalityStatus.visual === 'degraded'
                  ? 'bg-warning-amber'
                  : 'bg-outline'
              }`}
            />
            VISUAL: {modalityStatus.visual === 'active' ? 'OK' : modalityStatus.visual.toUpperCase()}
          </span>

          <span className={`px-3 py-1 rounded-full ${getAcousticPillStyle()} flex items-center gap-1.5 shadow-sm text-xs font-medium`}>
            <span
              className={`w-2 h-2 rounded-full ${
                modalityStatus.acoustic === 'active'
                  ? 'bg-success-green animate-pulse'
                  : modalityStatus.acoustic === 'degraded'
                  ? 'bg-warning-amber'
                  : 'bg-outline'
              }`}
            />
            ACOUSTIC: {modalityStatus.acoustic === 'degraded' ? 'NOISY' : modalityStatus.acoustic.toUpperCase()}
          </span>

          <span className="px-3 py-1 rounded-full pastel-blue flex items-center gap-1.5 shadow-sm text-xs font-medium">
            <span className="material-symbols-outlined text-[14px]">speed</span>
            {payloadLatencyMs}ms
          </span>
        </div>

        {/* Action Controls & Session Toggle */}
        <div className="flex items-center gap-3 border-l border-outline-variant pl-4 ml-1">
          {onSessionToggle && (
            <button
              onClick={onSessionToggle}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold font-data-label tracking-wide transition-all cursor-pointer flex items-center gap-1.5 ${
                isSessionActive
                  ? 'bg-alert-coral/10 text-alert-coral border border-alert-coral/30 hover:bg-alert-coral hover:text-white'
                  : 'bg-clinical-blue text-white hover:bg-blue-700 shadow-sm'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">
                {isSessionActive ? 'stop_circle' : 'play_circle'}
              </span>
              {isSessionActive ? 'Stop Stream' : 'Start Stream'}
            </button>
          )}

          <button
            title="Notifications"
            className="text-on-surface-variant hover:bg-surface-container-low transition-colors p-1.5 rounded-full cursor-pointer active:opacity-80 flex items-center justify-center"
          >
            <span className="material-symbols-outlined text-[20px]">notifications</span>
          </button>
          <button
            title="Settings"
            className="text-on-surface-variant hover:bg-surface-container-low transition-colors p-1.5 rounded-full cursor-pointer active:opacity-80 flex items-center justify-center"
          >
            <span className="material-symbols-outlined text-[20px]">settings</span>
          </button>
          <button
            title="Help"
            className="text-on-surface-variant hover:bg-surface-container-low transition-colors p-1.5 rounded-full cursor-pointer active:opacity-80 flex items-center justify-center"
          >
            <span className="material-symbols-outlined text-[20px]">help</span>
          </button>
        </div>
      </div>
    </header>
  );
});

