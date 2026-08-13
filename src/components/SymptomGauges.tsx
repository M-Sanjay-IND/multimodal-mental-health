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
    return <div className="h-6 w-full bg-[#0A0A0C] border border-[#1E1E24] rounded text-[9px] text-zinc-600 font-mono flex items-center justify-center">Accumulating 30s Trend...</div>;
  }

  const width = 140;
  const height = 24;
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

  const depPct = Math.min(100, Math.max(0, (continuousScores.depression / 34) * 100));
  const anxPct = Math.min(100, Math.max(0, (continuousScores.anxiety / 24) * 100));
  const strPct = Math.min(100, Math.max(0, (continuousScores.stress / 39) * 100));

  const margin = continuousScores.confidenceMargin || 1.5;

  return (
    <div className="bg-[#121216] border border-[#1E1E24] rounded p-4 flex flex-col justify-between h-full font-mono text-xs">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 border-b border-[#1E1E24] pb-2">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-zinc-200">Continuous Symptom Trends</span>
          {isPaused && (
            <span className="text-[10px] px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40">
              EPOCH PAUSED
            </span>
          )}
        </div>
        <span className="text-[10px] text-zinc-500">30s Moving Window Sparklines</span>
      </div>

      <div className="space-y-5 flex-1 flex flex-col justify-center">
        {/* Depression Meter & 30s Sparkline */}
        <div>
          <div className="flex justify-between items-baseline mb-1.5">
            <div className="flex items-center gap-2">
              <span className="text-zinc-200 font-semibold uppercase">Depression</span>
              <span className="text-zinc-500 text-[10px]">(0–34)</span>
            </div>
            <div className="flex items-center gap-3">
              <Sparkline data={depHistory} maxVal={34} colorHex="#FBBF24" />
              <div className="text-right">
                <span className="text-sm font-bold text-amber-400">
                  {continuousScores.depression.toFixed(1)}
                </span>
                <span className="text-[10px] text-zinc-400 block">±{margin.toFixed(1)}</span>
              </div>
            </div>
          </div>
          <div className="w-full h-1.5 bg-[#0A0A0C] border border-[#1E1E24] rounded relative overflow-hidden">
            <div
              className="h-full bg-amber-400 transition-all duration-500"
              style={{ width: `${depPct}%` }}
            />
          </div>
        </div>

        {/* Anxiety Meter & 30s Sparkline */}
        <div>
          <div className="flex justify-between items-baseline mb-1.5">
            <div className="flex items-center gap-2">
              <span className="text-zinc-200 font-semibold uppercase">Anxiety</span>
              <span className="text-zinc-500 text-[10px]">(0–24)</span>
            </div>
            <div className="flex items-center gap-3">
              <Sparkline data={anxHistory} maxVal={24} colorHex="#FB7185" />
              <div className="text-right">
                <span className="text-sm font-bold text-rose-400">
                  {continuousScores.anxiety.toFixed(1)}
                </span>
                <span className="text-[10px] text-zinc-400 block">±{margin.toFixed(1)}</span>
              </div>
            </div>
          </div>
          <div className="w-full h-1.5 bg-[#0A0A0C] border border-[#1E1E24] rounded relative overflow-hidden">
            <div
              className="h-full bg-rose-400 transition-all duration-500"
              style={{ width: `${anxPct}%` }}
            />
          </div>
        </div>

        {/* Stress Meter & 30s Sparkline */}
        <div>
          <div className="flex justify-between items-baseline mb-1.5">
            <div className="flex items-center gap-2">
              <span className="text-zinc-200 font-semibold uppercase">Stress</span>
              <span className="text-zinc-500 text-[10px]">(0–39)</span>
            </div>
            <div className="flex items-center gap-3">
              <Sparkline data={strHistory} maxVal={39} colorHex="#00FF66" />
              <div className="text-right">
                <span className="text-sm font-bold text-[#00FF66]">
                  {continuousScores.stress.toFixed(1)}
                </span>
                <span className="text-[10px] text-zinc-400 block">±{margin.toFixed(1)}</span>
              </div>
            </div>
          </div>
          <div className="w-full h-1.5 bg-[#0A0A0C] border border-[#1E1E24] rounded relative overflow-hidden">
            <div
              className="h-full bg-[#00FF66] transition-all duration-500"
              style={{ width: `${strPct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
});
