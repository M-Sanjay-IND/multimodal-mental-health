'use client';

import React, { memo } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

export const FastSHAPChart: React.FC = memo(function FastSHAPChart() {
  const { topRiskFactors, topResilienceFactors, featureOverrides, setFeatureOverride } =
    useDiagnosticResults();

  return (
    <div className="bg-surface-container-lowest rounded-[20px] pastel-shadow p-5 border border-border-subtle/50 font-sans flex flex-col justify-between h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="font-section-header text-section-header text-on-surface flex items-center gap-2">
          <span className="material-symbols-outlined text-clinical-blue text-[20px]">bar_chart</span>
          FastSHAP Feature Impact
        </h2>
        {Object.keys(featureOverrides).length > 0 && (
          <button
            onClick={() => {
              Object.keys(featureOverrides).forEach((k) => setFeatureOverride(k, null));
            }}
            className="text-[10px] font-data-mono text-clinical-blue hover:underline cursor-pointer"
          >
            Reset What-If
          </button>
        )}
      </div>

      <div className="space-y-3 font-caption text-caption">
        {/* Risk Contributors */}
        {topRiskFactors.slice(0, 3).map((factor) => {
          const currentOverride = featureOverrides[factor.featureName];
          const displayVal = currentOverride !== undefined ? currentOverride : factor.shapValue;
          const maxVal = 4.0;
          const pct = Math.min(50, Math.max(8, (displayVal / maxVal) * 50));

          return (
            <div key={factor.featureName} className="flex items-center gap-3">
              <span className="w-16 text-right text-on-surface-variant font-data-mono text-[11px] truncate" title={factor.featureName}>
                {factor.featureName.length > 8 ? factor.featureName.slice(0, 8) + '..' : factor.featureName}
              </span>
              <div className="flex-1 h-4 bg-surface-container-low rounded-sm relative group overflow-hidden">
                <div
                  className="absolute left-1/2 h-full dusty-rose rounded-r-sm opacity-90 transition-all duration-300"
                  style={{ width: `${pct}%` }}
                />
                <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-outline-variant" />
                <input
                  type="range"
                  min="0"
                  max="4"
                  step="0.1"
                  value={displayVal}
                  onChange={(e) => setFeatureOverride(factor.featureName, parseFloat(e.target.value))}
                  className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                  title={`Adjust ${factor.featureName} for What-If Analysis (+${displayVal.toFixed(2)} φ)`}
                />
              </div>
              <span className="w-12 font-data-mono text-[11px] text-alert-coral font-semibold">
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
              <span className="w-16 text-right text-on-surface-variant font-data-mono text-[11px] truncate" title={factor.featureName}>
                {factor.featureName.length > 8 ? factor.featureName.slice(0, 8) + '..' : factor.featureName}
              </span>
              <div className="flex-1 h-4 bg-surface-container-low rounded-sm relative overflow-hidden">
                <div
                  className="absolute right-1/2 h-full sage-green rounded-l-sm opacity-90 transition-all duration-300"
                  style={{ width: `${pct}%` }}
                />
                <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-outline-variant" />
              </div>
              <span className="w-12 font-data-mono text-[11px] text-success-green font-semibold">
                {factor.shapValue.toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
});

