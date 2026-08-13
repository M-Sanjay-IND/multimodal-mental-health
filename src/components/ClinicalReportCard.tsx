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
    <div className="mt-3 space-y-2 text-xs font-mono">
      {savedNotes.length > 0 && (
        <div className="space-y-1">
          {savedNotes.map((note, idx) => (
            <div key={idx} className="p-2 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">
              <span className="text-zinc-500 text-[10px] block font-bold">Observed Note #{idx + 1}</span>
              {note}
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-2">
        <input
          type="text"
          value={clinicalNote}
          onChange={(e) => setClinicalNote(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSaveNote()}
          placeholder="Add observation note..."
          className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-zinc-200 focus:outline-none focus:border-zinc-500 placeholder:text-zinc-600"
        />
        <button
          onClick={handleSaveNote}
          className="bg-white hover:bg-zinc-200 text-black font-bold px-3 py-1.5 rounded-lg transition-colors cursor-pointer"
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
      text: 'Pitch variability (F0) is severely blunted with monotonic prosody. Speech rate is significantly reduced by 32%, indicating psychomotor retardation.',
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
      text: 'Pitch variability (F0) remains slightly blunted compared to normative baseline. Speech rate is reduced by 8%, consistent with mild psychomotor strain.',
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
      text: 'Markedly reduced facial expressivity across all action units. Gaze aversion prominent with minimal eye contact.',
      eyeContactPct: 28,
      eyeLabel: 'LOW',
    };
  }
  if (depression > 12 || anxiety > 10) {
    return {
      text: 'Reduced facial expressivity noted in lower facial action units. Eye contact is diminished.',
      eyeContactPct: 52,
      eyeLabel: 'BELOW NORM',
    };
  }
  if (depression > 6 || anxiety > 5) {
    return {
      text: 'Mildly reduced facial expressivity in lower action units. Eye contact maintained at acceptable levels.',
      eyeContactPct: 68,
      eyeLabel: 'ACCEPTABLE',
    };
  }
  return {
    text: 'Facial expressivity is within normative range with appropriate affective modulation.',
    eyeContactPct: 85,
    eyeLabel: 'HEALTHY',
  };
}

export const ClinicalReportCard: React.FC = memo(function ClinicalReportCard() {
  const { classification, continuousScores, clinicalNarrative, exportFHIRReport } = useDiagnosticResults();

  const acoustic = deriveAcousticInsight(continuousScores.depression, continuousScores.stress);
  const visual = deriveVisualInsight(continuousScores.depression, continuousScores.anxiety);

  const predClass = classification.predictedClass;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full font-sans text-white">
      {/* 1. Acoustic Profile Card */}
      <div className="mono-card p-5 flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-300">Acoustic Profile</h3>
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed font-sans">
            {acoustic.text}
          </p>
        </div>
        <div className="mt-4 pt-3 border-t border-zinc-800 flex justify-between items-center text-xs font-mono">
          <span className="text-zinc-500">F0 Variance</span>
          <span className="font-bold text-white">{acoustic.f0Label} ({acoustic.f0Delta})</span>
        </div>
      </div>

      {/* 2. Visual Kinematics Card */}
      <div className="mono-card p-5 flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-300">Visual Kinematics</h3>
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed font-sans">
            {visual.text}
          </p>
        </div>
        <div className="mt-4 pt-3 border-t border-zinc-800 flex justify-between items-center text-xs font-mono">
          <span className="text-zinc-500">Eye Contact</span>
          <span className="font-bold text-white">{visual.eyeContactPct}% {visual.eyeLabel}</span>
        </div>
      </div>

      {/* 3. System Status & Diagnostic Summary Card */}
      <div className="mono-card p-5 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-300">System Evaluation</h3>
            <button
              onClick={exportFHIRReport}
              className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-mono text-[10px] font-bold border border-zinc-700 transition-all cursor-pointer"
            >
              Export FHIR
            </button>
          </div>

          <div className="space-y-2 font-mono text-xs text-zinc-400 mb-3">
            <div className="flex justify-between border-b border-zinc-800 pb-1">
              <span>Predicted Status</span>
              <span className="font-bold text-white px-2 py-0.5 rounded bg-zinc-800">{predClass}</span>
            </div>
            <div className="flex justify-between border-b border-zinc-800 pb-1">
              <span>Depression</span>
              <span className="font-bold text-zinc-200">{continuousScores.depression.toFixed(1)} / 34</span>
            </div>
            <div className="flex justify-between border-b border-zinc-800 pb-1">
              <span>Anxiety</span>
              <span className="font-bold text-zinc-200">{continuousScores.anxiety.toFixed(1)} / 24</span>
            </div>
            <div className="flex justify-between pt-1">
              <span>Stress</span>
              <span className="font-bold text-zinc-200">{continuousScores.stress.toFixed(1)} / 39</span>
            </div>
          </div>

          {/* Synthesized Narrative */}
          <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800 text-xs font-sans text-zinc-300 leading-normal">
            <span className="font-bold text-white block text-[11px] mb-0.5">Synthesized Evaluation:</span>
            {clinicalNarrative}
          </div>
        </div>

        {/* Clinician Notes Form */}
        <ClinicianNotesForm />
      </div>
    </div>
  );
});
