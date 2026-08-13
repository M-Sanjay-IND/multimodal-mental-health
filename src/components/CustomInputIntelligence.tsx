'use client';

import React, { useState, useCallback } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

interface FeatureMeta {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  defaultVal: number;
  description: string;
  category: 'behavioral' | 'visual' | 'acoustic' | 'physiological';
}

const FEATURE_DEFINITIONS: FeatureMeta[] = [
  // Behavioral & Lifestyle
  { key: 'Sleep_Quality', label: 'Sleep Quality Score', min: -3.0, max: 3.0, step: 0.1, unit: 'z-score', defaultVal: 1.2, description: 'Subjective & actigraphy sleep efficiency', category: 'behavioral' },
  { key: 'Social_Engagement', label: 'Social Engagement', min: 0, max: 100, step: 1, unit: '%', defaultVal: 75, description: 'Messaging & active social app interactions', category: 'behavioral' },
  { key: 'Daily_App_Usage_Min', label: 'Daily Screen Time', min: 0, max: 600, step: 5, unit: 'min', defaultVal: 150, description: 'Total active smartphone / desktop usage', category: 'behavioral' },
  { key: 'Typing_Speed_WPM', label: 'Typing Speed', min: 10, max: 120, step: 1, unit: 'WPM', defaultVal: 55, description: 'Keystroke velocity & motor dexterity', category: 'behavioral' },
  { key: 'Session_Frequency', label: 'Session Frequency', min: 1, max: 20, step: 1, unit: 'sessions/day', defaultVal: 6, description: 'Frequency of platform logins per day', category: 'behavioral' },
  { key: 'Idle_Time_Min', label: 'Screen Idle Duration', min: 0, max: 120, step: 2, unit: 'min', defaultVal: 15, description: 'Inactivity gaps between active sessions', category: 'behavioral' },

  // Visual Kinematics & Affective
  { key: 'Facial_Emotion_Variance', label: 'Facial Emotion Variance', min: 0.0, max: 1.0, step: 0.01, unit: 'index', defaultVal: 0.45, description: 'Dynamic expressivity across facial action units', category: 'visual' },
  { key: 'Eye_Blink_Rate', label: 'Eye Blink Frequency', min: 5, max: 60, step: 1, unit: 'blinks/min', defaultVal: 18, description: 'Spontaneous blink rate per minute', category: 'visual' },
  { key: 'Smile_Intensity', label: 'Smile Intensity (AU12)', min: 0.0, max: 1.0, step: 0.01, unit: 'index', defaultVal: 0.65, description: 'Zygomaticus major activation level', category: 'visual' },
  { key: 'Head_Motion_Index', label: 'Head Motion Index', min: 0.0, max: 1.0, step: 0.01, unit: 'index', defaultVal: 0.25, description: 'Postural stability and head movement range', category: 'visual' },

  // Acoustic & Speech
  { key: 'MFCC_Mean', label: 'Acoustic Timbre (MFCC Mean)', min: -2.0, max: 2.0, step: 0.05, unit: 'coeff', defaultVal: 0.15, description: 'Spectral envelope vocal vocalization mean', category: 'acoustic' },
  { key: 'MFCC_Variance', label: 'Acoustic Variance', min: 0.0, max: 5.0, step: 0.1, unit: 'variance', defaultVal: 1.2, description: 'Vocal resonance & energy dispersion', category: 'acoustic' },
  { key: 'Pitch_Mean', label: 'Fundamental Frequency (F0)', min: 50, max: 300, step: 5, unit: 'Hz', defaultVal: 180, description: 'Vocal pitch baseline frequency', category: 'acoustic' },
  { key: 'Speech_Rate', label: 'Speech Articulation Rate', min: 1.0, max: 5.0, step: 0.1, unit: 'words/sec', defaultVal: 3.2, description: 'Pacing of verbal communication', category: 'acoustic' },

  // Physiological Biomarkers
  { key: 'Heart_Rate_BPM', label: 'Resting Heart Rate', min: 50, max: 140, step: 1, unit: 'BPM', defaultVal: 72, description: 'Autonomic cardiovascular pulse rate', category: 'physiological' },
  { key: 'HRV_Index', label: 'Heart Rate Variability (HRV)', min: 10, max: 120, step: 1, unit: 'ms', defaultVal: 65, description: 'RMSSD parasympathetic autonomic tone', category: 'physiological' },
  { key: 'Skin_Temperature', label: 'Peripheral Skin Temp', min: 34.0, max: 39.0, step: 0.1, unit: '°C', defaultVal: 36.6, description: 'Distal vasomotor temperature response', category: 'physiological' },
  { key: 'GSR_Level', label: 'Galvanic Skin Response (GSR)', min: 0.1, max: 10.0, step: 0.1, unit: 'μS', defaultVal: 1.4, description: 'Electrodermal sympathetic arousal level', category: 'physiological' },
];

const PRESETS = [
  {
    name: 'Optimal / Healthy Baseline',
    values: {
      Sleep_Quality: 2.1,
      Social_Engagement: 85,
      Daily_App_Usage_Min: 120,
      Typing_Speed_WPM: 65,
      Session_Frequency: 5,
      Idle_Time_Min: 10,
      Facial_Emotion_Variance: 0.65,
      Eye_Blink_Rate: 16,
      Smile_Intensity: 0.75,
      Head_Motion_Index: 0.35,
      MFCC_Mean: 0.2,
      MFCC_Variance: 1.8,
      Pitch_Mean: 195,
      Speech_Rate: 3.5,
      Heart_Rate_BPM: 65,
      HRV_Index: 78,
      Skin_Temperature: 36.6,
      GSR_Level: 0.8,
    },
  },
  {
    name: 'Mild Work Stress & Fatigue',
    values: {
      Sleep_Quality: 0.2,
      Social_Engagement: 60,
      Daily_App_Usage_Min: 280,
      Typing_Speed_WPM: 50,
      Session_Frequency: 8,
      Idle_Time_Min: 25,
      Facial_Emotion_Variance: 0.40,
      Eye_Blink_Rate: 22,
      Smile_Intensity: 0.45,
      Head_Motion_Index: 0.25,
      MFCC_Mean: -0.1,
      MFCC_Variance: 1.1,
      Pitch_Mean: 175,
      Speech_Rate: 2.8,
      Heart_Rate_BPM: 78,
      HRV_Index: 52,
      Skin_Temperature: 36.5,
      GSR_Level: 2.4,
    },
  },
  {
    name: 'Moderate Depressive Affect',
    values: {
      Sleep_Quality: -1.8,
      Social_Engagement: 30,
      Daily_App_Usage_Min: 420,
      Typing_Speed_WPM: 32,
      Session_Frequency: 12,
      Idle_Time_Min: 55,
      Facial_Emotion_Variance: 0.18,
      Eye_Blink_Rate: 28,
      Smile_Intensity: 0.15,
      Head_Motion_Index: 0.12,
      MFCC_Mean: -0.8,
      MFCC_Variance: 0.5,
      Pitch_Mean: 145,
      Speech_Rate: 2.1,
      Heart_Rate_BPM: 85,
      HRV_Index: 32,
      Skin_Temperature: 36.1,
      GSR_Level: 4.8,
    },
  },
  {
    name: 'Severe Crisis / High Agitation',
    values: {
      Sleep_Quality: -2.8,
      Social_Engagement: 10,
      Daily_App_Usage_Min: 550,
      Typing_Speed_WPM: 22,
      Session_Frequency: 16,
      Idle_Time_Min: 85,
      Facial_Emotion_Variance: 0.05,
      Eye_Blink_Rate: 38,
      Smile_Intensity: 0.02,
      Head_Motion_Index: 0.05,
      MFCC_Mean: -1.5,
      MFCC_Variance: 0.2,
      Pitch_Mean: 120,
      Speech_Rate: 1.4,
      Heart_Rate_BPM: 112,
      HRV_Index: 18,
      Skin_Temperature: 35.8,
      GSR_Level: 8.5,
    },
  },
];

export const CustomInputIntelligence: React.FC = function CustomInputIntelligence() {
  const { injectUploadedPayload, classification, continuousScores, clinicalNarrative } = useDiagnosticResults();

  const [featureValues, setFeatureValues] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    FEATURE_DEFINITIONS.forEach((f) => {
      initial[f.key] = f.defaultVal;
    });
    return initial;
  });

  const [activeCategory, setActiveCategory] = useState<'all' | 'behavioral' | 'visual' | 'acoustic' | 'physiological'>('all');
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [lastEvaluatedPreset, setLastEvaluatedPreset] = useState<string | null>(null);

  const handleSliderChange = (key: string, val: number) => {
    setFeatureValues((prev) => ({ ...prev, [key]: val }));
    setLastEvaluatedPreset(null);
  };

  const handleApplyPreset = (preset: typeof PRESETS[0]) => {
    setFeatureValues(preset.values);
    setLastEvaluatedPreset(preset.name);
  };

  const handleEvaluateCustomInputs = useCallback(async () => {
    setIsEvaluating(true);

    const tabularVector = FEATURE_DEFINITIONS.map((f) => featureValues[f.key] ?? f.defaultVal);

    // Compute realistic visual & acoustic vectors derived from custom inputs
    const sleep = featureValues['Sleep_Quality'] ?? 0;
    const emotionVar = featureValues['Facial_Emotion_Variance'] ?? 0.5;
    const speechRate = featureValues['Speech_Rate'] ?? 3.0;

    const visualVector = new Array(128).fill(0).map((_, idx) => (idx % 2 === 0 ? emotionVar : 1 - emotionVar));
    const acousticVector = new Array(256).fill(0).map((_, idx) => (idx % 3 === 0 ? speechRate / 5 : (sleep + 3) / 6));

    try {
      await injectUploadedPayload({
        tabularVector,
        visualVector,
        acousticVector,
        presetName: lastEvaluatedPreset || undefined,
      });
    } finally {
      setIsEvaluating(false);
    }
  }, [featureValues, injectUploadedPayload, lastEvaluatedPreset]);

  const categories = [
    { id: 'all', label: 'All Inputs (18)' },
    { id: 'behavioral', label: 'Behavioral & Lifestyle' },
    { id: 'visual', label: 'Visual Kinematics' },
    { id: 'acoustic', label: 'Acoustic & Speech' },
    { id: 'physiological', label: 'Physiological Biomarkers' },
  ] as const;

  const filteredFeatures = activeCategory === 'all'
    ? FEATURE_DEFINITIONS
    : FEATURE_DEFINITIONS.filter((f) => f.category === activeCategory);

  const predClass = classification.predictedClass;
  const statusBadgeColor =
    predClass === 'Healthy' ? 'bg-emerald-100 text-emerald-800 border-emerald-300' :
    predClass === 'Mild' ? 'bg-amber-100 text-amber-800 border-amber-300' :
    predClass === 'Moderate' ? 'bg-orange-100 text-orange-800 border-orange-300' :
    'bg-rose-100 text-rose-800 border-rose-300';

  return (
    <div className="space-y-8 w-full font-sans">
      {/* Header & Description */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
        <div>
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            Value Inputs Intelligence & Simulation
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl">
            Configure raw biomarker inputs across Behavioral, Visual, Acoustic, and Physiological streams. The DCMF-Net fusion model processes your exact input values to generate real-time multi-task diagnostic intelligence.
          </p>
        </div>

        <button
          onClick={handleEvaluateCustomInputs}
          disabled={isEvaluating}
          className="px-6 py-3 text-white rounded-xl font-bold text-sm shadow-md transition-all cursor-pointer disabled:opacity-60 flex items-center justify-center gap-2 shrink-0"
          style={{
            background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
          }}
        >
          {isEvaluating ? 'Evaluating Model...' : 'Run Intelligence Evaluation'}
        </button>
      </div>

      {/* Benchmark Presets */}
      <div>
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">
          Clinical Clinical Benchmark Presets
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {PRESETS.map((preset) => (
            <button
              key={preset.name}
              onClick={() => {
                handleApplyPreset(preset);
              }}
              className={`p-3.5 rounded-xl border text-left transition-all cursor-pointer ${
                lastEvaluatedPreset === preset.name
                  ? 'bg-blue-50 border-blue-400 ring-2 ring-blue-200'
                  : 'bg-white hover:bg-slate-50 border-slate-200'
              }`}
            >
              <div className="font-bold text-xs text-slate-800">{preset.name}</div>
              <div className="text-[11px] text-slate-500 mt-1 font-mono">
                Sleep: {preset.values.Sleep_Quality} • HRV: {preset.values.HRV_Index}ms
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Category Filter Tabs */}
      <div className="flex gap-2 border-b border-slate-200 pb-2 overflow-x-auto">
        {categories.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setActiveCategory(cat.id)}
            className={`px-4 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
              activeCategory === cat.id
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            {cat.label}
          </button>
        ))}
      </div>

      {/* Input Controls Grid & Live Intelligence Output Side-by-Side */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 Cols: Category Input Sliders */}
        <div className="lg:col-span-7 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {filteredFeatures.map((feat) => {
              const val = featureValues[feat.key] ?? feat.defaultVal;
              return (
                <div key={feat.key} className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-semibold text-slate-800">{feat.label}</span>
                    <span className="font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
                      {val} {feat.unit}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={feat.min}
                    max={feat.max}
                    step={feat.step}
                    value={val}
                    onChange={(e) => handleSliderChange(feat.key, parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                    <span>{feat.min}</span>
                    <span className="text-[10px] text-slate-500 font-sans italic truncate max-w-[140px]">
                      {feat.description}
                    </span>
                    <span>{feat.max}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right 5 Cols: Live Intelligence Results */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm space-y-5 sticky top-6">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <h4 className="font-bold text-sm text-slate-800">
                Intelligence Assessment Output
              </h4>
              <span className={`px-3 py-1 rounded-full text-xs font-bold border ${statusBadgeColor}`}>
                {predClass}
              </span>
            </div>

            {/* Continuous Severity Gauges */}
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs font-medium text-slate-600 mb-1">
                  <span>Depression (PHQ-9 Proxy)</span>
                  <span className="font-mono font-bold text-slate-900">
                    {continuousScores.depression.toFixed(1)} / 34
                  </span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full transition-all duration-300"
                    style={{ width: `${(continuousScores.depression / 34) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-medium text-slate-600 mb-1">
                  <span>Anxiety (GAD-7 Proxy)</span>
                  <span className="font-mono font-bold text-slate-900">
                    {continuousScores.anxiety.toFixed(1)} / 24
                  </span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-rose-500 rounded-full transition-all duration-300"
                    style={{ width: `${(continuousScores.anxiety / 24) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-medium text-slate-600 mb-1">
                  <span>Stress (PSS Scale)</span>
                  <span className="font-mono font-bold text-slate-900">
                    {continuousScores.stress.toFixed(1)} / 39
                  </span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full transition-all duration-300"
                    style={{ width: `${(continuousScores.stress / 39) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Synthesized Clinical Narrative */}
            <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-700 leading-relaxed">
              <span className="font-bold text-slate-900 block mb-1">DCMF-Net Narrative Synthesis:</span>
              {clinicalNarrative}
            </div>

            <button
              onClick={handleEvaluateCustomInputs}
              className="w-full py-2.5 bg-slate-900 hover:bg-black text-white rounded-xl text-xs font-bold transition-all cursor-pointer"
            >
              Re-Calculate Intelligence
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
