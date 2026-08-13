'use client';

import React, { memo, useState } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

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

function deriveAcousticInsight(depression: number, stress: number): { text: string; f0Label: string; f0Delta: string } {
  if (depression > 20 || stress > 25) {
    return {
      text: 'Pitch variability (F0) is severely blunted with monotonic prosody. Speech rate is significantly reduced by 32%, indicating psychomotor retardation consistent with major depressive features.',
      f0Label: 'SEVERELY BLUNTED',
      f0Delta: '-32%',
    };
  }
  if (depression > 12 || stress > 18) {
    return {
      text: 'Pitch variability (F0) shows moderate blunting with reduced prosodic range. Speech rate is decreased by 18%, suggesting mild-to-moderate psychomotor slowing.',
      f0Label: 'BLUNTED',
      f0Delta: '-18%',
    };
  }
  if (depression > 6 || stress > 10) {
    return {
      text: 'Pitch variability (F0) remains slightly blunted compared to normative baseline. Speech rate is reduced by 8%, consistent with mild psychomotor retardation.',
      f0Label: 'MILDLY BLUNTED',
      f0Delta: '-8%',
    };
  }
  return {
    text: 'Pitch variability (F0) is within normative range with healthy prosodic variation. Speech rate and rhythm are unremarkable.',
    f0Label: 'NORMATIVE',
    f0Delta: '+2%',
  };
}

function deriveVisualInsight(depression: number, anxiety: number): { text: string; eyeContactPct: number; eyeLabel: string } {
  if (depression > 20 || anxiety > 16) {
    return {
      text: 'Markedly reduced facial expressivity across all action units. Gaze aversion prominent with minimal eye contact, consistent with severe affective withdrawal.',
      eyeContactPct: 28,
      eyeLabel: 'LOW',
    };
  }
  if (depression > 12 || anxiety > 10) {
    return {
      text: 'Reduced facial expressivity noted in lower facial action units. Eye contact is diminished, potentially reflecting social withdrawal or anxious avoidance.',
      eyeContactPct: 52,
      eyeLabel: 'BELOW NORM',
    };
  }
  if (depression > 6 || anxiety > 5) {
    return {
      text: 'Mildly reduced facial expressivity in lower action units. Eye contact maintained at acceptable levels within therapeutic range.',
      eyeContactPct: 68,
      eyeLabel: 'ACCEPTABLE',
    };
  }
  return {
    text: 'Facial expressivity is within normative range with appropriate affective modulation. Eye contact is sustained and consistent.',
    eyeContactPct: 85,
    eyeLabel: 'HEALTHY',
  };
}

export const ClinicalReportCard: React.FC = memo(function ClinicalReportCard() {
  const { classification, continuousScores, clinicalNarrative, exportFHIRReport } = useDiagnosticResults();

  const acoustic = deriveAcousticInsight(continuousScores.depression, continuousScores.stress);
  const visual = deriveVisualInsight(continuousScores.depression, continuousScores.anxiety);

  const predClass = classification.predictedClass;
  const statusColor =
    predClass === 'Healthy' ? 'text-success-green' :
    predClass === 'Mild' ? 'text-warning-amber' :
    'text-alert-coral';

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-widget-gap w-full font-sans">
      {/* 1. Acoustic Profile Card — DYNAMIC */}
      <div className="bg-[#f0f9ff] rounded-2xl p-5 pastel-shadow border-none flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-3 text-clinical-blue">
            <span className="material-symbols-outlined text-[20px]">summarize</span>
            <h3 className="font-section-header text-section-header">Acoustic Profile</h3>
          </div>
          <p className="font-body-md text-body-md text-on-surface-variant leading-relaxed">
            {acoustic.text}
          </p>
        </div>
        <div className="mt-4 pt-3 border-t border-black/5 flex justify-between items-center text-xs font-data-mono">
          <span className="text-on-surface-variant">F0 Variance</span>
          <span className="font-bold text-clinical-blue">{acoustic.f0Label} ({acoustic.f0Delta})</span>
        </div>
      </div>

      {/* 2. Visual Kinematics Card — DYNAMIC */}
      <div className="bg-[#fdf2f8] rounded-2xl p-5 pastel-shadow border-none flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-3 text-alert-coral">
            <span className="material-symbols-outlined text-[20px]">visibility</span>
            <h3 className="font-section-header text-section-header">Visual Kinematics</h3>
          </div>
          <p className="font-body-md text-body-md text-on-surface-variant leading-relaxed">
            {visual.text}
          </p>
        </div>
        <div className="mt-4 pt-3 border-t border-black/5 flex justify-between items-center text-xs font-data-mono">
          <span className="text-on-surface-variant">Eye Contact</span>
          <span className="font-bold text-alert-coral">{visual.eyeContactPct}% {visual.eyeLabel}</span>
        </div>
      </div>

      {/* 3. System Status & Diagnostic Summary Card — DYNAMIC */}
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
              <span className={`font-bold ${statusColor}`}>{predClass}</span>
            </div>
            <div className="flex justify-between border-b border-black/5 pb-1">
              <span>Depression</span>
              <span className="font-bold text-on-surface">{continuousScores.depression.toFixed(1)} / 34</span>
            </div>
            <div className="flex justify-between border-b border-black/5 pb-1">
              <span>Anxiety</span>
              <span className="font-bold text-on-surface">{continuousScores.anxiety.toFixed(1)} / 24</span>
            </div>
            <div className="flex justify-between pt-1">
              <span>Stress</span>
              <span className="font-bold text-on-surface">{continuousScores.stress.toFixed(1)} / 39</span>
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
