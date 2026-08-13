'use client';

import React, { memo } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

export const FastSHAPChart: React.FC = memo(function FastSHAPChart() {
  const { topRiskFactors, topResilienceFactors, featureOverrides, setFeatureOverride } =
    useDiagnosticResults();

  return (
    <div className="bg-[#121216] border border-[#1E1E24] rounded p-4 flex flex-col h-full font-mono text-xs">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 border-b border-[#1E1E24] pb-2">
        <div>
          <span className="font-semibold text-zinc-200 block">FastSHAP Feature Attributions</span>
          <span className="text-[10px] text-zinc-500">Interactive What-If Inspector</span>
        </div>
        <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          Shapley (φ)
        </span>
      </div>

      <div className="flex-1 flex flex-col justify-center space-y-4">
        {/* Risk Indicators (+phi growing right) */}
        <div className="space-y-3">
          <span className="text-[10px] text-emerald-400 uppercase tracking-wider block font-semibold">
            Risk Contributors (+ϕ_i) — Drag Slider to Test Intervention
          </span>
          {topRiskFactors.map((factor) => {
            const currentOverride = featureOverrides[factor.featureName];
            const displayVal = currentOverride !== undefined ? currentOverride : factor.shapValue;
            const maxVal = 4.0;
            const pct = Math.min(50, Math.max(5, (displayVal / maxVal) * 50));

            return (
              <div key={factor.featureName} className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-zinc-300 flex items-center gap-1.5">
                    {factor.featureName}
                    {currentOverride !== undefined && (
                      <span className="text-[9px] px-1 bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 rounded">
                        WHAT-IF
                      </span>
                    )}
                  </span>
                  <span className="text-emerald-400 font-semibold">
                    +{displayVal.toFixed(2)} φ
                  </span>
                </div>
                <div className="w-full h-2 bg-[#0A0A0C] border border-[#1E1E24] rounded relative flex items-center">
                  <div
                    className="absolute h-full bg-emerald-400 rounded-r transition-all duration-300"
                    style={{ left: '50%', width: `${pct}%` }}
                  />
                  <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-zinc-600" />
                  <input
                    type="range"
                    min="0"
                    max="4"
                    step="0.1"
                    value={displayVal}
                    onChange={(e) => setFeatureOverride(factor.featureName, parseFloat(e.target.value))}
                    className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                    title={`Adjust ${factor.featureName} for What-If Analysis`}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Protective Markers (-phi growing left) */}
        <div className="space-y-3 pt-3 border-t border-[#1E1E24] border-dashed">
          <span className="text-[10px] text-rose-400 uppercase tracking-wider block font-semibold">
            Resilience Markers (-ϕ_i)
          </span>
          {topResilienceFactors.map((factor) => {
            const maxVal = 4.0;
            const absVal = Math.abs(factor.shapValue);
            const pct = Math.min(50, Math.max(5, (absVal / maxVal) * 50));

            return (
              <div key={factor.featureName}>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-zinc-300">{factor.featureName}</span>
                  <span className="text-rose-400 font-semibold">
                    {factor.shapValue.toFixed(2)} φ
                  </span>
                </div>
                <div className="w-full h-1.5 bg-[#0A0A0C] border border-[#1E1E24] rounded relative">
                  <div
                    className="absolute h-full bg-rose-400 rounded-l transition-all duration-500"
                    style={{ right: '50%', width: `${pct}%` }}
                  />
                  <div className="absolute left-1/2 top-[-2px] bottom-[-2px] w-[1px] bg-zinc-600" />
                </div>
              </div>
            );
          })}
        </div>

        {Object.keys(featureOverrides).length > 0 && (
          <button
            onClick={() => {
              Object.keys(featureOverrides).forEach((k) => setFeatureOverride(k, null));
            }}
            className="w-full py-1 text-[10px] font-mono text-cyan-400 hover:underline border border-cyan-500/30 rounded bg-cyan-500/10 cursor-pointer"
          >
            Reset What-If Interventions
          </button>
        )}
      </div>
    </div>
  );
});
