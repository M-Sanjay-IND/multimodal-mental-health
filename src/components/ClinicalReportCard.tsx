'use client';

import React, { memo, useState } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

/**
 * Isolated Clinician Notes Form Component.
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
    <div className="mt-3 space-y-2 font-body-md text-xs">
      {savedNotes.length > 0 && (
        <div className="space-y-1 font-data-mono text-[11px]">
          {savedNotes.map((note, idx) => (
            <div key={idx} className="p-2 rounded-lg bg-white/80 border border-success-green/30 text-on-surface">
              <span className="text-success-green text-[10px] block font-bold">Observed #{idx + 1}</span>
              {note}
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-2 font-body-md text-xs">
        <input
          type="text"
          value={clinicalNote}
          onChange={(e) => setClinicalNote(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSaveNote()}
          placeholder="Add observation note..."
          className="flex-1 bg-white/80 border border-outline-variant rounded-lg px-3 py-1.5 text-on-surface focus:outline-none focus:border-clinical-blue placeholder:text-on-surface-variant/60"
        />
        <button
          onClick={handleSaveNote}
          className="bg-clinical-blue hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg font-bold transition-colors cursor-pointer"
        >
          Save
        </button>
      </div>
    </div>
  );
});

export const ClinicalReportCard: React.FC = memo(function ClinicalReportCard() {
  const { classification, clinicalNarrative, exportFHIRReport } = useDiagnosticResults();

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-widget-gap w-full font-sans">
      {/* 1. Acoustic Profile Card */}
      <div className="bg-[#f0f9ff] rounded-2xl p-5 pastel-shadow border-none flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-3 text-clinical-blue">
            <span className="material-symbols-outlined text-[20px]">summarize</span>
            <h3 className="font-section-header text-section-header">Acoustic Profile</h3>
          </div>
          <p className="font-body-md text-body-md text-on-surface-variant leading-relaxed">
            Pitch variability (F0) remains blunted compared to normative baseline. Speech rate is reduced by 15%, consistent with mild psychomotor retardation.
          </p>
        </div>
        <div className="mt-4 pt-3 border-t border-black/5 flex justify-between items-center text-xs font-data-mono">
          <span className="text-on-surface-variant">F0 Variance</span>
          <span className="font-bold text-clinical-blue">BLUNTED (-15%)</span>
        </div>
      </div>

      {/* 2. Visual Kinematics Card */}
      <div className="bg-[#fdf2f8] rounded-2xl p-5 pastel-shadow border-none flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-3 text-alert-coral">
            <span className="material-symbols-outlined text-[20px]">visibility</span>
            <h3 className="font-section-header text-section-header">Visual Kinematics</h3>
          </div>
          <p className="font-body-md text-body-md text-on-surface-variant leading-relaxed">
            Reduced facial expressivity noted in lower facial action units. Eye contact maintained at 68% of session, within acceptable therapeutic range.
          </p>
        </div>
        <div className="mt-4 pt-3 border-t border-black/5 flex justify-between items-center text-xs font-data-mono">
          <span className="text-on-surface-variant">Eye Contact</span>
          <span className="font-bold text-alert-coral">68% ACCEPTABLE</span>
        </div>
      </div>

      {/* 3. System Status & Diagnostic Summary Card */}
      <div className="bg-[#f0fdf4] rounded-2xl p-5 pastel-shadow border-none flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-success-green">
              <span className="material-symbols-outlined text-[20px]">check_circle</span>
              <h3 className="font-section-header text-section-header">System Status</h3>
            </div>
            <button
              onClick={exportFHIRReport}
              className="px-2.5 py-1 rounded-lg bg-success-green/20 text-success-green hover:bg-success-green hover:text-white font-data-mono text-[10px] font-bold transition-all cursor-pointer border border-success-green/30"
            >
              Export FHIR
            </button>
          </div>

          <div className="space-y-2 font-data-mono text-data-mono text-on-surface-variant mb-3">
            <div className="flex justify-between border-b border-black/5 pb-1">
              <span>Predicted Status</span>
              <span className="font-bold text-success-green">{classification.predictedClass}</span>
            </div>
            <div className="flex justify-between border-b border-black/5 pb-1">
              <span>Data Quality</span>
              <span className="font-bold text-on-surface">HIGH_FIDELITY</span>
            </div>
            <div className="flex justify-between pt-1">
              <span>Uptime</span>
              <span className="font-bold text-on-surface">99.98%</span>
            </div>
          </div>

          {/* Synthesized Narrative */}
          <div className="p-2.5 rounded-lg bg-white/70 border border-success-green/20 text-xs font-body-md text-on-surface-variant leading-normal">
            <span className="font-bold text-on-surface block text-[11px] mb-0.5">Synthesized Evaluation:</span>
            {clinicalNarrative}
          </div>
        </div>

        {/* Clinician Notes Form */}
        <ClinicianNotesForm />
      </div>
    </div>
  );
});

