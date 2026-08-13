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
  category: 'Behavioral & Lifestyle' | 'Visual Kinematics' | 'Acoustic & Speech' | 'Physiological Biomarkers';
}

const FEATURE_DEFINITIONS: FeatureMeta[] = [
  // Behavioral & Lifestyle
  { key: 'Sleep_Quality', label: 'Sleep Quality (1-5)', min: 1.0, max: 5.0, step: 0.1, unit: 'score', defaultVal: 3.5, description: 'Subjective & actigraphy sleep score', category: 'Behavioral & Lifestyle' },
  { key: 'Social_Engagement', label: 'Social Engagement (1-5)', min: 1.0, max: 5.0, step: 0.1, unit: 'score', defaultVal: 4.0, description: 'Active social interaction level', category: 'Behavioral & Lifestyle' },
  { key: 'Daily_App_Usage_Min', label: 'Daily Screen Time', min: 0, max: 720, step: 5, unit: 'min', defaultVal: 150, description: 'Smartphone/desktop screen duration', category: 'Behavioral & Lifestyle' },
  { key: 'Typing_Speed_WPM', label: 'Typing Speed', min: 0, max: 150, step: 1, unit: 'WPM', defaultVal: 55, description: 'Keystroke velocity & motor control', category: 'Behavioral & Lifestyle' },
  { key: 'Session_Frequency', label: 'Session Frequency', min: 0, max: 30, step: 1, unit: 'sessions/day', defaultVal: 6, description: 'Daily active logins count', category: 'Behavioral & Lifestyle' },
  { key: 'Idle_Time_Min', label: 'Screen Idle Duration', min: 0, max: 180, step: 1, unit: 'min', defaultVal: 15, description: 'Inactivity duration between actions', category: 'Behavioral & Lifestyle' },

  // Visual Kinematics
  { key: 'Facial_Emotion_Variance', label: 'Facial Emotion Variance', min: 0.0, max: 1.0, step: 0.01, unit: 'index', defaultVal: 0.45, description: 'Expressivity across facial action units', category: 'Visual Kinematics' },
  { key: 'Eye_Blink_Rate', label: 'Eye Blink Frequency', min: 0, max: 80, step: 1, unit: 'blinks/min', defaultVal: 18, description: 'Blinks per minute rate', category: 'Visual Kinematics' },
  { key: 'Smile_Intensity', label: 'Smile Intensity (AU12)', min: 0.0, max: 1.0, step: 0.01, unit: 'index', defaultVal: 0.65, description: 'Zygomaticus major contraction', category: 'Visual Kinematics' },
  { key: 'Head_Motion_Index', label: 'Head Motion Index', min: 0.0, max: 1.0, step: 0.01, unit: 'index', defaultVal: 0.25, description: 'Head movement stability index', category: 'Visual Kinematics' },

  // Acoustic & Speech
  { key: 'MFCC_Mean', label: 'MFCC Mean', min: -5.0, max: 5.0, step: 0.05, unit: 'val', defaultVal: 0.15, description: 'Mel-frequency cepstral coefficient mean', category: 'Acoustic & Speech' },
  { key: 'MFCC_Variance', label: 'MFCC Variance', min: 0.0, max: 10.0, step: 0.1, unit: 'var', defaultVal: 1.2, description: 'Spectral energy dispersion', category: 'Acoustic & Speech' },
  { key: 'Pitch_Mean', label: 'Pitch Mean (F0)', min: 40, max: 350, step: 1, unit: 'Hz', defaultVal: 180, description: 'Vocal pitch baseline frequency', category: 'Acoustic & Speech' },
  { key: 'Speech_Rate', label: 'Speech Rate', min: 0.0, max: 10.0, step: 0.1, unit: 'words/sec', defaultVal: 3.2, description: 'Verbal articulation speed', category: 'Acoustic & Speech' },

  // Physiological Biomarkers
  { key: 'Heart_Rate_BPM', label: 'Resting Heart Rate', min: 40, max: 180, step: 1, unit: 'BPM', defaultVal: 72, description: 'Cardiovascular heart rate', category: 'Physiological Biomarkers' },
  { key: 'HRV_Index', label: 'Heart Rate Variability (HRV)', min: 5, max: 150, step: 1, unit: 'ms', defaultVal: 65, description: 'Parasympathetic RMSSD autonomic index', category: 'Physiological Biomarkers' },
  { key: 'Skin_Temperature', label: 'Skin Temperature', min: 30.0, max: 42.0, step: 0.1, unit: '°C', defaultVal: 36.6, description: 'Vasomotor peripheral temperature', category: 'Physiological Biomarkers' },
  { key: 'GSR_Level', label: 'Galvanic Skin Response (GSR)', min: 0.0, max: 20.0, step: 0.1, unit: 'μS', defaultVal: 1.4, description: 'Electrodermal sympathetic arousal', category: 'Physiological Biomarkers' },
];

const PRESETS: Record<string, Record<string, number>> = {
  'Healthy / Normal': {
    Sleep_Quality: 4.5,
    Social_Engagement: 4.5,
    Daily_App_Usage_Min: 110,
    Typing_Speed_WPM: 65,
    Session_Frequency: 4,
    Idle_Time_Min: 10,
    Facial_Emotion_Variance: 0.65,
    Eye_Blink_Rate: 15,
    Smile_Intensity: 0.75,
    Head_Motion_Index: 0.35,
    MFCC_Mean: 0.2,
    MFCC_Variance: 1.8,
    Pitch_Mean: 195,
    Speech_Rate: 3.5,
    Heart_Rate_BPM: 68,
    HRV_Index: 75,
    Skin_Temperature: 36.6,
    GSR_Level: 0.8,
  },
  'Mild Stress / Fatigue': {
    Sleep_Quality: 3.0,
    Social_Engagement: 3.2,
    Daily_App_Usage_Min: 250,
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
  'Moderate Distress': {
    Sleep_Quality: 2.0,
    Social_Engagement: 2.0,
    Daily_App_Usage_Min: 420,
    Typing_Speed_WPM: 35,
    Session_Frequency: 12,
    Idle_Time_Min: 55,
    Facial_Emotion_Variance: 0.20,
    Eye_Blink_Rate: 28,
    Smile_Intensity: 0.20,
    Head_Motion_Index: 0.15,
    MFCC_Mean: -0.8,
    MFCC_Variance: 0.5,
    Pitch_Mean: 150,
    Speech_Rate: 2.1,
    Heart_Rate_BPM: 88,
    HRV_Index: 32,
    Skin_Temperature: 36.2,
    GSR_Level: 4.8,
  },
  'Severe Crisis': {
    Sleep_Quality: 1.0,
    Social_Engagement: 1.0,
    Daily_App_Usage_Min: 580,
    Typing_Speed_WPM: 20,
    Session_Frequency: 18,
    Idle_Time_Min: 90,
    Facial_Emotion_Variance: 0.05,
    Eye_Blink_Rate: 42,
    Smile_Intensity: 0.05,
    Head_Motion_Index: 0.05,
    MFCC_Mean: -1.6,
    MFCC_Variance: 0.2,
    Pitch_Mean: 120,
    Speech_Rate: 1.4,
    Heart_Rate_BPM: 115,
    HRV_Index: 16,
    Skin_Temperature: 35.8,
    GSR_Level: 8.5,
  },
};

export const CustomInputIntelligence: React.FC = function CustomInputIntelligence() {
  const { injectUploadedPayload } = useDiagnosticResults();

  const [formData, setFormData] = useState<Record<string, number>>(() => {
    const init: Record<string, number> = {};
    FEATURE_DEFINITIONS.forEach((f) => {
      init[f.key] = f.defaultVal;
    });
    return init;
  });

  const [isLoading, setIsLoading] = useState(false);
  const [modelResponse, setModelResponse] = useState<any | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string>('');

  const handleInputChange = (key: string, rawVal: string) => {
    const val = parseFloat(rawVal);
    if (!isNaN(val)) {
      setFormData((prev) => ({ ...prev, [key]: val }));
    } else {
      setFormData((prev) => ({ ...prev, [key]: 0 }));
    }
    setSelectedPreset('');
  };

  const handlePresetSelect = (presetName: string) => {
    if (PRESETS[presetName]) {
      setFormData(PRESETS[presetName]);
      setSelectedPreset(presetName);
    }
  };

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsLoading(true);
    setErrorMessage(null);

    // Build exact 18-dim tabular vector from form values
    const tabularValues = FEATURE_DEFINITIONS.map((f) => formData[f.key] ?? f.defaultVal);

    // Neutral 128D visual and 256D acoustic vectors for tabular-first inference
    const visualValues = new Array(128).fill(0.0);
    const acousticValues = new Array(256).fill(0.0);

    const payload = {
      visual_vector: { values: visualValues },
      acoustic_vector: { values: acousticValues },
      tabular: { values: tabularValues },
    };

    try {
      // 1. Direct REST call to PyTorch model server
      const res = await fetch('http://localhost:8000/evaluate/rest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const data = await res.json();
      setModelResponse(data);

      // 2. Also inject into central DiagnosticContext so global state updates
      await injectUploadedPayload({
        tabularVector: tabularValues,
        visualVector: visualValues,
        acousticVector: acousticValues,
      });
    } catch (err: any) {
      setErrorMessage(
        err?.message || 'Failed to connect to model server at http://localhost:8000/evaluate/rest'
      );
    } finally {
      setIsLoading(false);
    }
  }, [formData, injectUploadedPayload]);

  const categories = [
    'Behavioral & Lifestyle',
    'Visual Kinematics',
    'Acoustic & Speech',
    'Physiological Biomarkers',
  ] as const;

  return (
    <div className="space-y-8 w-full font-sans">
      {/* Top Banner */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900">
            Model Value Input Form
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl">
            Directly input value parameters into the 18 feature slots of the PyTorch DCMF-Net model. Submitting sends the raw tabular payload directly to the model endpoint for inference.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <select
            value={selectedPreset}
            onChange={(e) => handlePresetSelect(e.target.value)}
            className="px-3 py-2 bg-slate-50 border border-slate-300 rounded-xl text-xs font-semibold text-slate-700 focus:outline-none cursor-pointer"
          >
            <option value="">Load Preset Values...</option>
            {Object.keys(PRESETS).map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>

          <button
            onClick={() => handleSubmit()}
            disabled={isLoading}
            className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-bold shadow-md transition-all cursor-pointer disabled:opacity-50"
          >
            {isLoading ? 'Running Model...' : 'Submit Values to Model'}
          </button>
        </div>
      </div>

      {errorMessage && (
        <div className="p-4 bg-rose-50 border border-rose-200 text-rose-800 rounded-xl text-xs font-medium">
          {errorMessage}
        </div>
      )}

      <form onSubmit={handleSubmit} className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Input Form Fields (Col 7) */}
        <div className="lg:col-span-7 space-y-6">
          {categories.map((cat) => {
            const features = FEATURE_DEFINITIONS.filter((f) => f.category === cat);
            return (
              <div key={cat} className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider border-b border-slate-100 pb-2">
                  {cat}
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {features.map((f) => {
                    const currentVal = formData[f.key] ?? f.defaultVal;
                    return (
                      <div key={f.key} className="space-y-1.5">
                        <div className="flex justify-between items-center text-xs">
                          <label className="font-semibold text-slate-700">{f.label}</label>
                          <span className="text-[10px] text-slate-400 font-mono">[{f.unit}]</span>
                        </div>
                        <input
                          type="number"
                          min={f.min}
                          max={f.max}
                          step={f.step}
                          value={currentVal}
                          onChange={(e) => handleInputChange(f.key, e.target.value)}
                          className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs font-mono text-slate-900 focus:bg-white focus:border-blue-500 focus:outline-none transition-all"
                        />
                        <p className="text-[10px] text-slate-400 italic leading-tight">
                          {f.description}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs rounded-xl shadow-md transition-all cursor-pointer disabled:opacity-50"
          >
            {isLoading ? 'Processing Model Forward Pass...' : 'Run PyTorch DCMF-Net Inference'}
          </button>
        </div>

        {/* Model Output Results Panel (Col 5) */}
        <div className="lg:col-span-5">
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-6 sticky top-6">
            <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-3 flex items-center justify-between">
              <span>Model Inference Output</span>
              {modelResponse && (
                <span className="text-[10px] font-mono text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded font-semibold">
                  REST HTTP 200 OK
                </span>
              )}
            </h3>

            {!modelResponse ? (
              <div className="text-center py-12 text-slate-400 text-xs space-y-2">
                <div className="font-semibold text-slate-500">No output generated yet</div>
                <p>Fill out the input values or select a preset and click "Submit Values to Model".</p>
              </div>
            ) : (
              <div className="space-y-6 text-xs">
                {/* 1. Classification Output */}
                <div className="space-y-3">
                  <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                    Categorical Classification
                  </div>

                  <div className="flex items-center justify-between bg-slate-50 p-3 rounded-xl border border-slate-200">
                    <span className="font-medium text-slate-700">Predicted Severity Class</span>
                    <span className="font-bold font-mono text-sm px-3 py-1 bg-blue-600 text-white rounded-lg">
                      {modelResponse.classification?.predicted_class || 'Healthy'}
                    </span>
                  </div>

                  {/* Class Probabilities Bars */}
                  {modelResponse.classification?.probabilities && (
                    <div className="space-y-2 pt-1">
                      <div className="text-[10px] font-bold text-slate-500">Softmax Probabilities:</div>
                      {Object.entries(modelResponse.classification.probabilities).map(([clsName, prob]: [string, any]) => {
                        const pct = Math.round(prob * 100);
                        return (
                          <div key={clsName} className="space-y-1">
                            <div className="flex justify-between text-[11px] font-mono text-slate-600">
                              <span>{clsName}</span>
                              <span>{pct}%</span>
                            </div>
                            <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-blue-500 rounded-full transition-all duration-500"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* 2. Regression Symptom Scores */}
                {modelResponse.regression && (
                  <div className="space-y-3 border-t border-slate-100 pt-4">
                    <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                      Continuous Symptom Regression
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-center font-mono">
                      <div className="bg-emerald-50 p-2.5 rounded-xl border border-emerald-100">
                        <span className="text-[10px] text-emerald-700 block font-semibold">Depression</span>
                        <span className="text-sm font-bold text-emerald-900">
                          {modelResponse.regression.depression_score}
                        </span>
                        <span className="text-[9px] text-emerald-600 block">/ 34</span>
                      </div>

                      <div className="bg-amber-50 p-2.5 rounded-xl border border-amber-100">
                        <span className="text-[10px] text-amber-700 block font-semibold">Anxiety</span>
                        <span className="text-sm font-bold text-amber-900">
                          {modelResponse.regression.anxiety_score}
                        </span>
                        <span className="text-[9px] text-amber-600 block">/ 24</span>
                      </div>

                      <div className="bg-rose-50 p-2.5 rounded-xl border border-rose-100">
                        <span className="text-[10px] text-rose-700 block font-semibold">Stress</span>
                        <span className="text-sm font-bold text-rose-900">
                          {modelResponse.regression.stress_score}
                        </span>
                        <span className="text-[9px] text-rose-600 block">/ 39</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* 3. FastSHAP Feature Attributions */}
                {modelResponse.shap_attribution?.attributions && (
                  <div className="border-t border-slate-100 pt-4 space-y-2">
                    <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                      FastSHAP Feature Impact
                    </div>
                    <div className="space-y-1.5 font-mono text-[11px]">
                      {Object.entries(modelResponse.shap_attribution.attributions)
                        .slice(0, 6)
                        .map(([featName, val]: [string, any]) => {
                          const isRisk = val > 0;
                          return (
                            <div key={featName} className="flex justify-between items-center bg-slate-50 px-2.5 py-1.5 rounded border border-slate-100">
                              <span className="text-slate-700 font-sans">{featName.replace(/_/g, ' ')}</span>
                              <span className={`font-bold ${isRisk ? 'text-rose-600' : 'text-emerald-600'}`}>
                                {isRisk ? `+${val}` : val}
                              </span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                )}

                {/* 4. Clinical Narrative */}
                {modelResponse.narrative && (
                  <div className="border-t border-slate-100 pt-4 space-y-1.5">
                    <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                      Clinical Narrative Synthesis
                    </div>
                    <p className="p-3 bg-slate-50 border border-slate-200 rounded-xl text-slate-700 text-xs leading-relaxed font-sans">
                      {modelResponse.narrative}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </form>
    </div>
  );
};
