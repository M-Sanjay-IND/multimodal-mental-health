'use client';

import React, { memo } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

export const FastSHAPChart: React.FC = memo(function FastSHAPChart() {
  const { topRiskFactors, topResilienceFactors, featureOverrides, setFeatureOverride } =
    useDiagnosticResults();

  return (
    <div className="mono-card p-5 font-sans text-white flex flex-col justify-between h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-bold tracking-wide uppercase text-zinc-200">
          FastSHAP Feature Impact
        </h2>
        {Object.keys(featureOverrides).length > 0 && (
          <button
            onClick={() => {
              Object.keys(featureOverrides).forEach((k) => setFeatureOverride(k, null));
            }}
            className="text-[10px] font-mono text-zinc-400 hover:text-white underline cursor-pointer"
          >
            Reset
          </button>
        )}
      </div>

      <div className="space-y-3 font-mono text-xs">
        {/* Risk Contributors */}
        {topRiskFactors.slice(0, 3).map((factor) => {
          const currentOverride = featureOverrides[factor.featureName];
          const displayVal = currentOverride !== undefined ? currentOverride : factor.shapValue;
          const maxVal = 4.0;
          const pct = Math.min(50, Math.max(8, (displayVal / maxVal) * 50));

          return (
            <div key={factor.featureName} className="flex items-center gap-3">
              <span className="w-20 text-right text-zinc-400 text-[11px] truncate" title={factor.featureName}>
                {factor.featureName.replace(/_/g, ' ')}
              </span>
              <div className="flex-1 h-3 bg-zinc-900 rounded relative overflow-hidden border border-zinc-800">
                <div
                  className="absolute left-1/2 h-full bg-white transition-all duration-300"
                  style={{ width: `${pct}%` }}
                />
                <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-zinc-700" />
                <input
                  type="range"
                  min="0"
                  max="4"
                  step="0.1"
                  value={displayVal}
                  onChange={(e) => setFeatureOverride(factor.featureName, parseFloat(e.target.value))}
                  className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                />
              </div>
              <span className="w-12 text-[11px] text-white font-bold">
                +{displayVal.toFixed(1)}
              </span>
            </div>
          );
        })}

        {/* Protective Factors */}
        {topResilienceFactors.slice(0, 2).map((factor) => {
          const maxVal = 4.0;
          const absVal = Math.abs(factor.shapValue);
          const pct = Math.min(50, Math.max(8, (absVal / maxVal) * 50));

          return (
            <div key={factor.featureName} className="flex items-center gap-3">
              <span className="w-20 text-right text-zinc-400 text-[11px] truncate" title={factor.featureName}>
                {factor.featureName.replace(/_/g, ' ')}
              </span>
              <div className="flex-1 h-3 bg-zinc-900 rounded relative overflow-hidden border border-zinc-800">
                <div
                  className="absolute right-1/2 h-full bg-zinc-400 transition-all duration-300"
                  style={{ width: `${pct}%` }}
                />
                <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-zinc-700" />
              </div>
              <span className="w-12 text-[11px] text-zinc-300 font-bold">
                {factor.shapValue.toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
});
