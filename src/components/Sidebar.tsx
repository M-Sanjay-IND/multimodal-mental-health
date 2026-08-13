'use client';

import React, { memo, useState } from 'react';

export interface SidebarProps {
  onSessionToggle?: () => void;
  isSessionActive?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = memo(function Sidebar({
  onSessionToggle,
  isSessionActive = false,
}) {
  const [activeTab, setActiveTab] = useState('Dashboard');

  return (
    <nav className="hidden md:flex flex-col py-6 px-4 bg-surface-container-lowest text-clinical-blue font-body-md fixed left-0 top-0 h-full w-60 shadow-sm border-r border-outline-variant z-40 overflow-y-auto">
      {/* Clinician Profile Avatar */}
      <div className="mb-6 flex flex-col items-center text-center">
        <div className="w-14 h-14 rounded-full bg-surface-container-low mb-2 overflow-hidden shadow-sm border border-outline-variant">
          <img
            className="w-full h-full object-cover"
            alt="Dr. Adrian Sterling profile"
            src="https://images.unsplash.com/photo-1559839734-2b71ea197ec2?q=80&w=200&auto=format&fit=crop"
          />
        </div>
        <div className="font-nav-brand text-sm font-bold text-clinical-blue">
          Dr. Adrian Sterling
        </div>
        <div className="font-caption text-[11px] text-on-surface-variant uppercase tracking-wider">
          Senior Psychiatrist
        </div>
      </div>

      {/* Primary Navigation Links */}
      <div className="space-y-1 mb-6">
        {[
          { label: 'Dashboard', icon: 'dashboard' },
          { label: 'Patient Monitoring', icon: 'videocam' },
          { label: 'Session Analysis', icon: 'psychology' },
          { label: 'Reports', icon: 'description' },
        ].map((item) => (
          <button
            key={item.label}
            onClick={() => setActiveTab(item.label)}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-all duration-200 ease-in-out cursor-pointer font-semibold text-xs ${
              activeTab === item.label
                ? 'bg-secondary-fixed text-on-secondary-fixed-variant shadow-sm'
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
            }`}
          >
            <span
              className={`material-symbols-outlined text-[20px] ${
                activeTab === item.label ? 'text-clinical-blue' : ''
              }`}
            >
              {item.icon}
            </span>
            {item.label}
          </button>
        ))}
      </div>

      {/* Session Quick Control */}
      <div className="mt-auto pt-4 border-t border-outline-variant">
        {onSessionToggle && (
          <button
            onClick={onSessionToggle}
            className={`w-full py-2.5 px-4 rounded-xl font-bold text-xs shadow-sm transition-all cursor-pointer flex items-center justify-center gap-2 ${
              isSessionActive
                ? 'bg-alert-coral text-white hover:bg-rose-700'
                : 'bg-clinical-blue text-white hover:bg-blue-700'
            }`}
          >
            <span className="material-symbols-outlined text-[18px]">
              {isSessionActive ? 'stop_circle' : 'play_circle'}
            </span>
            {isSessionActive ? 'Stop Session' : 'Start Session'}
          </button>
        )}
      </div>
    </nav>
  );
});
