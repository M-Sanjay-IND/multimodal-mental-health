'use client';

import React, { memo } from 'react';
import { ScoreHistoryPoint, useDiagnosticResults } from '../hooks/useDiagnosticResults';

interface SparklineProps {
  data: number[];
  maxVal: number;
}

const Sparkline: React.FC<SparklineProps> = memo(function Sparkline({ data, maxVal }) {
  if (data.length < 2) {
    return (
      <div className="h-4 w-16 bg-zinc-900 rounded text-[9px] text-zinc-500 font-mono flex items-center justify-center">
        Trend...
      </div>
    );
  }

  const width = 64;
  const height = 16;
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
        stroke="#ffffff"
        strokeWidth="1.2"
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
    <div className="mono-card p-5 flex-1 flex flex-col justify-between h-full font-sans text-white">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-bold tracking-wide uppercase text-zinc-200">
          Symptom Severity Gauges
        </h2>
        {isPaused && (
          <span className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 text-[10px] font-mono border border-zinc-700">
            PAUSED
          </span>
        )}
      </div>

      <div className="space-y-5 flex-1 flex flex-col justify-center font-mono">
        {/* Depression */}
        <div>
          <div className="flex justify-between items-center text-xs mb-1.5 text-zinc-300">
            <span>Depression (PHQ-9)</span>
            <div className="flex items-center gap-3">
              <Sparkline data={depHistory} maxVal={34} />
              <span className="font-bold text-white">{continuousScores.depression.toFixed(1)} / 34</span>
            </div>
          </div>
          <div className="h-2 w-full bg-zinc-900 rounded-full overflow-hidden border border-zinc-800">
            <div
              className="h-full bg-white rounded-full transition-all duration-500"
              style={{ width: `${depPct}%` }}
            />
          </div>
        </div>

        {/* Anxiety */}
        <div>
          <div className="flex justify-between items-center text-xs mb-1.5 text-zinc-300">
            <span>Anxiety (GAD-7)</span>
            <div className="flex items-center gap-3">
              <Sparkline data={anxHistory} maxVal={24} />
              <span className="font-bold text-white">{continuousScores.anxiety.toFixed(1)} / 24</span>
            </div>
          </div>
          <div className="h-2 w-full bg-zinc-900 rounded-full overflow-hidden border border-zinc-800">
            <div
              className="h-full bg-zinc-400 rounded-full transition-all duration-500"
              style={{ width: `${anxPct}%` }}
            />
          </div>
        </div>

        {/* Stress */}
        <div>
          <div className="flex justify-between items-center text-xs mb-1.5 text-zinc-300">
            <span>Stress (PSS Scale)</span>
            <div className="flex items-center gap-3">
              <Sparkline data={strHistory} maxVal={39} />
              <span className="font-bold text-white">{continuousScores.stress.toFixed(1)} / 39</span>
            </div>
          </div>
          <div className="h-2 w-full bg-zinc-900 rounded-full overflow-hidden border border-zinc-800">
            <div
              className="h-full bg-zinc-200 rounded-full transition-all duration-500"
              style={{ width: `${strPct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
});
