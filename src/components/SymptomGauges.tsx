'use client';

import React, { memo } from 'react';
import { ScoreHistoryPoint, useDiagnosticResults } from '../hooks/useDiagnosticResults';

interface SparklineProps {
  data: number[];
  maxVal: number;
  colorHex: string;
}

const Sparkline: React.FC<SparklineProps> = memo(function Sparkline({ data, maxVal, colorHex }) {
  if (data.length < 2) {
    return (
      <div className="h-5 w-20 bg-surface-container-low rounded text-[9px] text-on-surface-variant font-data-mono flex items-center justify-center">
        30s Trend...
      </div>
    );
  }

  const width = 80;
  const height = 20;
  const points = data
    .map((val, idx) => {
      const x = (idx / (data.length - 1)) * width;
      const y = height - (Math.min(maxVal, Math.max(0, val)) / maxVal) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline
        fill="none"
        stroke={colorHex}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
});

export const SymptomGauges: React.FC = memo(function SymptomGauges() {
  const { continuousScores, historyScores, isPaused } = useDiagnosticResults();

  const depHistory = historyScores.map((p: ScoreHistoryPoint) => p.depression);
  const anxHistory = historyScores.map((p: ScoreHistoryPoint) => p.anxiety);
  const strHistory = historyScores.map((p: ScoreHistoryPoint) => p.stress);

  const depPct = Math.min(100, Math.max(0, Math.round((continuousScores.depression / 34) * 100)));
  const anxPct = Math.min(100, Math.max(0, Math.round((continuousScores.anxiety / 24) * 100)));
  const strPct = Math.min(100, Math.max(0, Math.round((continuousScores.stress / 39) * 100)));

  return (
    <div className="bg-surface-container-lowest rounded-[20px] pastel-shadow p-5 flex-1 border border-border-subtle/50 flex flex-col justify-between h-full font-sans">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="font-section-header text-section-header text-on-surface flex items-center gap-2">
          <span className="material-symbols-outlined text-clinical-blue text-[20px]">trending_up</span>
          Severity Indicators
        </h2>
        {isPaused && (
          <span className="px-2 py-0.5 rounded-full bg-warning-amber/20 text-warning-amber text-[10px] font-bold font-data-mono">
            EPOCH PAUSED
          </span>
        )}
      </div>

      <div className="space-y-5 flex-1 flex flex-col justify-center">
        {/* Indicator 1: Anhedonia (Depression) */}
        <div>
          <div className="flex justify-between items-center font-data-label text-data-label mb-2 text-on-surface">
            <span className="flex items-center gap-1.5 font-semibold">
              Anhedonia <span className="text-[10px] text-on-surface-variant font-normal">(Depression)</span>
            </span>
            <div className="flex items-center gap-3">
              <Sparkline data={depHistory} maxVal={34} colorHex="#4b5563" />
              <span className="font-data-mono font-bold text-on-surface">{depPct}%</span>
            </div>
          </div>
          <div className="relative h-3 w-full pastel-progress-bg rounded-full overflow-visible">
            <div
              className="absolute top-0 left-0 h-full rounded-full sage-green transition-all duration-500 shadow-sm"
              style={{ width: `${depPct}%` }}
            />
            {/* Confidence Interval Halo */}
            <div
              className="absolute top-1/2 -translate-y-1/2 h-5 rounded-full sage-green opacity-25 blur-sm pointer-events-none transition-all duration-500"
              style={{
                left: `${Math.max(0, depPct - 12)}%`,
                width: '24%',
              }}
            />
            {/* Normative marker */}
            <div className="absolute top-[-2px] bottom-[-2px] w-[2px] bg-white left-[50%] z-10 shadow-sm" />
          </div>
        </div>

        {/* Indicator 2: Psychomotor Agitation (Anxiety) */}
        <div>
          <div className="flex justify-between items-center font-data-label text-data-label mb-2 text-on-surface">
            <span className="flex items-center gap-1.5 font-semibold">
              Psychomotor Agitation <span className="text-[10px] text-on-surface-variant font-normal">(Anxiety)</span>
            </span>
            <div className="flex items-center gap-3">
              <Sparkline data={anxHistory} maxVal={24} colorHex="#f43f5e" />
              <span className="font-data-mono font-bold text-on-surface">{anxPct}%</span>
            </div>
          </div>
          <div className="relative h-3 w-full pastel-progress-bg rounded-full overflow-visible">
            <div
              className="absolute top-0 left-0 h-full rounded-full dusty-rose transition-all duration-500 shadow-sm"
              style={{ width: `${anxPct}%` }}
            />
            {/* Confidence Interval Halo */}
            <div
              className="absolute top-1/2 -translate-y-1/2 h-5 rounded-full dusty-rose opacity-25 blur-sm pointer-events-none transition-all duration-500"
              style={{
                left: `${Math.max(0, anxPct - 10)}%`,
                width: '20%',
              }}
            />
            {/* Normative marker */}
            <div className="absolute top-[-2px] bottom-[-2px] w-[2px] bg-white left-[30%] z-10 shadow-sm" />
          </div>
        </div>

        {/* Indicator 3: Speech Latency (Stress) */}
        <div>
          <div className="flex justify-between items-center font-data-label text-data-label mb-2 text-on-surface">
            <span className="flex items-center gap-1.5 font-semibold">
              Speech Latency <span className="text-[10px] text-on-surface-variant font-normal">(Stress)</span>
            </span>
            <div className="flex items-center gap-3">
              <Sparkline data={strHistory} maxVal={39} colorHex="#2563eb" />
              <span className="font-data-mono font-bold text-on-surface">{strPct}%</span>
            </div>
          </div>
          <div className="relative h-3 w-full pastel-progress-bg rounded-full overflow-visible">
            <div
              className="absolute top-0 left-0 h-full rounded-full bg-clinical-blue/60 transition-all duration-500 shadow-sm"
              style={{ width: `${strPct}%` }}
            />
            {/* Normative marker */}
            <div className="absolute top-[-2px] bottom-[-2px] w-[2px] bg-white left-[60%] z-10 shadow-sm" />
          </div>
        </div>
      </div>
    </div>
  );
});

