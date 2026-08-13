'use client';

import React, { memo, useEffect, useState } from 'react';

export interface BaselineCalibrationModalProps {
  isOpen: boolean;
  onComplete: () => void;
  onCancel: () => void;
}

export const BaselineCalibrationModal: React.FC<BaselineCalibrationModalProps> = memo(
  function BaselineCalibrationModal({ isOpen, onComplete, onCancel }) {
    const [secondsRemaining, setSecondsRemaining] = useState(15);

    useEffect(() => {
      if (!isOpen) {
        setSecondsRemaining(15);
        return;
      }

      const timer = setInterval(() => {
        setSecondsRemaining((prev) => {
          if (prev <= 1) {
            clearInterval(timer);
            onComplete();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => clearInterval(timer);
    }, [isOpen, onComplete]);

    if (!isOpen) return null;

    const progressPct = ((15 - secondsRemaining) / 15) * 100;

    return (
      <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 font-mono">
        <div className="bg-[#121216] border border-[#1E1E24] rounded-lg max-w-md w-full p-6 space-y-5 text-zinc-100 shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-[#1E1E24] pb-3">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
              <h3 className="text-sm font-semibold tracking-tight text-white">
                15-Second Baseline Calibration
              </h3>
            </div>
            <span className="text-xs text-emerald-400 font-bold font-mono">
              {secondsRemaining}s Remaining
            </span>
          </div>

          {/* Instructions */}
          <div className="space-y-3 text-xs text-zinc-300 leading-relaxed font-sans">
            <p>
              Please instruct the patient to look neutrally at the camera without speaking or making strong facial expressions.
            </p>
            <div className="p-3 rounded bg-[#0A0A0C] border border-[#1E1E24] text-[11px] text-zinc-400 font-mono space-y-1">
              <div className="text-emerald-400 font-semibold">Recording Baseline Metrics:</div>
              <div>• Resting Eye Aspect Ratio (EAR)</div>
              <div>• Baseline Fundamental Pitch (F0)</div>
              <div>• Baseline Galvanic Skin Response (GSR)</div>
            </div>
          </div>

          {/* Progress Meter */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-[11px] font-mono text-zinc-400">
              <span>Calibrating Patient Baseline...</span>
              <span>{Math.round(progressPct)}%</span>
            </div>
            <div className="w-full h-2 bg-[#0A0A0C] border border-[#1E1E24] rounded overflow-hidden">
              <div
                className="h-full bg-emerald-400 transition-all duration-1000 ease-linear"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-end gap-2 pt-2 border-t border-[#1E1E24]">
            <button
              onClick={onCancel}
              className="px-3 py-1.5 rounded border border-[#1E1E24] bg-[#0A0A0C] text-zinc-400 hover:text-zinc-200 text-xs transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              onClick={onComplete}
              className="px-4 py-1.5 rounded bg-emerald-400 text-black font-bold text-xs hover:bg-emerald-300 transition-colors cursor-pointer"
            >
              Skip &amp; Use Default
            </button>
          </div>
        </div>
      </div>
    );
  }
);
