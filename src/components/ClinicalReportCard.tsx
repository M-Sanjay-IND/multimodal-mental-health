'use client';

import React, { memo, useState } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

/**
 * Isolated Clinician Notes Form Component.
 * Detached from high-frequency (100ms) WebSocket state updates to eliminate typing lag.
 */
const ClinicianNotesForm: React.FC = memo(function ClinicianNotesForm() {
  const [clinicalNote, setClinicalNote] = useState('');
  const [savedNotes, setSavedNotes] = useState<string[]>([]);

  const handleSaveNote = () => {
    if (clinicalNote.trim()) {
      setSavedNotes((prev) => [...prev, clinicalNote.trim()]);
      setClinicalNote('');
    }
  };

  return (
    <div className="mt-auto space-y-3 font-mono text-xs">
      {/* Saved Clinician Notes */}
      {savedNotes.length > 0 && (
        <div className="space-y-1.5 font-mono text-[11px]">
          {savedNotes.map((note, idx) => (
            <div key={idx} className="p-2 rounded bg-[#0A0A0C] border border-[#00FF66]/30 text-zinc-200">
              <span className="text-[#00FF66] text-[10px] block font-semibold">Note #{idx + 1}</span>
              {note}
            </div>
          ))}
        </div>
      )}

      {/* Input Box */}
      <div className="flex gap-2 font-mono text-xs">
        <input
          type="text"
          value={clinicalNote}
          onChange={(e) => setClinicalNote(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSaveNote()}
          placeholder="Add clinician observation note..."
          className="flex-1 bg-[#0A0A0C] border border-[#1E1E24] rounded px-3 py-1.5 text-zinc-200 focus:outline-none focus:border-[#00FF66]/50 placeholder:text-zinc-600"
        />
        <button
          onClick={handleSaveNote}
          className="bg-[#1E1E24] hover:bg-zinc-800 border border-zinc-700 px-4 py-1.5 rounded text-zinc-200 font-semibold transition-colors cursor-pointer"
        >
          Save
        </button>
      </div>
    </div>
  );
});

export const ClinicalReportCard: React.FC = memo(function ClinicalReportCard() {
  const { classification, clinicalNarrative, exportFHIRReport } = useDiagnosticResults();

  const getStatusBadgeStyle = (status: string) => {
    switch (status) {
      case 'Healthy':
        return 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400';
      case 'Mild':
        return 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400';
      case 'Moderate':
        return 'bg-amber-500/10 border-amber-500/30 text-amber-400';
      case 'Severe':
      default:
        return 'bg-rose-500/10 border-rose-500/30 text-rose-400';
    }
  };

  return (
    <div className="bg-[#121216] border border-[#1E1E24] rounded p-4 flex flex-col justify-between h-full font-sans">
      {/* Header with 1-Click FHIR Export */}
      <div className="flex items-center justify-between mb-3 border-b border-[#1E1E24] pb-2 font-mono text-xs">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-zinc-200">Clinical Evaluation Summary</span>
          <span className={`px-2.5 py-0.5 rounded border font-semibold text-[10px] uppercase ${getStatusBadgeStyle(classification.predictedClass)}`}>
            Status: {classification.predictedClass}
          </span>
        </div>
        <button
          onClick={exportFHIRReport}
          className="px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 text-[10px] font-mono font-semibold transition-colors cursor-pointer"
        >
          Export FHIR Report
        </button>
      </div>

      {/* Categorical Distribution */}
      <div className="mb-3 space-y-1 font-mono text-[11px]">
        <span className="text-zinc-400 text-[10px] uppercase tracking-wider block">
          Diagnostic Class Probabilities
        </span>
        <div className="grid grid-cols-4 gap-2">
          {[
            { label: 'Healthy', val: classification.healthy },
            { label: 'Mild', val: classification.mild },
            { label: 'Moderate', val: classification.moderate },
            { label: 'Severe', val: classification.severe },
          ].map((item) => (
            <div key={item.label} className="p-2 rounded bg-[#0A0A0C] border border-[#1E1E24] text-center">
              <span className="text-zinc-500 text-[10px] block">{item.label}</span>
              <span className="text-zinc-200 font-semibold text-xs font-mono">
                {(item.val * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Synthesized Rule Narrative */}
      <div className="p-3 rounded bg-[#0A0A0C] border border-[#1E1E24] font-mono text-xs text-zinc-300 leading-relaxed mb-3">
        <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold block mb-1">
          Synthesized Evaluation
        </span>
        <p>{clinicalNarrative}</p>
      </div>

      {/* Contextual Clinical Decision Support Systems (CDSS) Nudge */}
      <div className="p-2.5 rounded bg-cyan-500/10 border border-cyan-500/30 font-mono text-[11px] text-cyan-200 mb-3 space-y-1">
        <div className="text-[10px] text-cyan-400 font-semibold uppercase tracking-wider flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
          CDSS Nudge — Clinical Decision Support
        </div>
        <p className="leading-normal">
          High Galvanic Skin Response (GSR_Level) coupled with speech rate reduction (+1.4 risk attribution) detected.
          <span className="text-white font-semibold block mt-0.5">Suggested Prompt: Ask patient about recent acute sleep quality or stressors.</span>
        </p>
      </div>

      {/* Detached Clinician Notes Form */}
      <ClinicianNotesForm />
    </div>
  );
});
