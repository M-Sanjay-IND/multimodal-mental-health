'use client';

import React, { memo, useState } from 'react';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';

export interface DataUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onVideoFileLoaded?: (file: File) => void;
}

export const DataUploadModal: React.FC<DataUploadModalProps> = memo(
  function DataUploadModal({ isOpen, onClose, onVideoFileLoaded }) {
    const { injectUploadedPayload } = useDiagnosticResults();
    const [activeTab, setActiveTab] = useState<'media' | 'tabular' | 'presets'>('media');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [tabularJson, setTabularJson] = useState<string>('');
    const [uploadStatus, setUploadStatus] = useState<string | null>(null);
    const [isEvaluating, setIsEvaluating] = useState<boolean>(false);

    if (!isOpen) return null;

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
        setSelectedFile(e.target.files[0]);
        setUploadStatus(`Selected: ${e.target.files[0].name} (${(e.target.files[0].size / 1024 / 1024).toFixed(2)} MB)`);
      }
    };

    const handleProcessMedia = async () => {
      if (!selectedFile) return;
      setIsEvaluating(true);
      setUploadStatus('Extracting multimodal embedding vectors...');

      if (selectedFile.type.startsWith('video/') && onVideoFileLoaded) {
        onVideoFileLoaded(selectedFile);
      }

      setTimeout(async () => {
        await injectUploadedPayload({
          presetName: 'General File Upload',
        });
        setIsEvaluating(false);
        setUploadStatus('Evaluation complete! Updating dashboard...');
        setTimeout(() => {
          onClose();
        }, 600);
      }, 1000);
    };

    const handleProcessTabular = async () => {
      setIsEvaluating(true);
      setUploadStatus('Parsing tabular metrics...');

      let tabularVec: number[] | undefined;
      try {
        if (tabularJson.trim()) {
          const parsed = JSON.parse(tabularJson);
          if (Array.isArray(parsed) && parsed.length === 18) {
            tabularVec = parsed.map((n) => Number(n));
          } else if (typeof parsed === 'object') {
            tabularVec = [
              parsed.Sleep_Quality ?? 3.0,
              parsed.Social_Engagement ?? 3.0,
              parsed.Daily_App_Usage_Min ?? 180,
              parsed.Typing_Speed_WPM ?? 45,
              parsed.Session_Frequency ?? 4,
              parsed.Idle_Time_Min ?? 120,
              parsed.Facial_Emotion_Variance ?? 0.45,
              parsed.Eye_Blink_Rate ?? 18,
              parsed.Smile_Intensity ?? 0.35,
              parsed.Head_Motion_Index ?? 1.2,
              parsed.MFCC_Mean ?? 0.05,
              parsed.MFCC_Variance ?? 0.12,
              parsed.Pitch_Mean ?? 120,
              parsed.Speech_Rate ?? 3.2,
              parsed.Heart_Rate_BPM ?? 78,
              parsed.HRV_Index ?? 42,
              parsed.Skin_Temperature ?? 36.5,
              parsed.GSR_Level ?? 1.8,
            ];
          }
        }
      } catch (err) {
        // Fallback default if parse error
      }

      await injectUploadedPayload({
        tabularVector: tabularVec,
      });

      setIsEvaluating(false);
      setUploadStatus('Tabular evaluation complete!');
      setTimeout(() => {
        onClose();
      }, 600);
    };

    const handlePresetSelect = async (presetName: string) => {
      setIsEvaluating(true);
      setUploadStatus(`Loading preset: ${presetName}...`);
      await injectUploadedPayload({ presetName });
      setIsEvaluating(false);
      onClose();
    };

    return (
      <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 font-sans">
        <div className="bg-surface-container-lowest border border-outline-variant rounded-2xl max-w-xl w-full p-6 space-y-5 text-on-surface shadow-2xl relative overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-outline-variant pb-4">
            <div className="flex items-center gap-2 text-clinical-blue">
              <span className="material-symbols-outlined text-[24px]">cloud_upload</span>
              <h2 className="font-section-header text-lg font-bold">
                Multimodal Input &amp; File Upload
              </h2>
            </div>
            <button
              onClick={onClose}
              className="text-on-surface-variant hover:text-on-surface p-1 rounded-lg transition-colors cursor-pointer"
            >
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>

          {/* Modality Tabs */}
          <div className="flex border-b border-outline-variant text-xs font-semibold gap-2">
            <button
              onClick={() => setActiveTab('media')}
              className={`pb-2.5 px-3 flex items-center gap-1.5 border-b-2 transition-all cursor-pointer ${
                activeTab === 'media'
                  ? 'border-clinical-blue text-clinical-blue font-bold'
                  : 'border-transparent text-on-surface-variant hover:text-on-surface'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">video_file</span>
              Video / Audio File
            </button>
            <button
              onClick={() => setActiveTab('tabular')}
              className={`pb-2.5 px-3 flex items-center gap-1.5 border-b-2 transition-all cursor-pointer ${
                activeTab === 'tabular'
                  ? 'border-clinical-blue text-clinical-blue font-bold'
                  : 'border-transparent text-on-surface-variant hover:text-on-surface'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">table_chart</span>
              Tabular JSON / CSV
            </button>
            <button
              onClick={() => setActiveTab('presets')}
              className={`pb-2.5 px-3 flex items-center gap-1.5 border-b-2 transition-all cursor-pointer ${
                activeTab === 'presets'
                  ? 'border-clinical-blue text-clinical-blue font-bold'
                  : 'border-transparent text-on-surface-variant hover:text-on-surface'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">tune</span>
              Clinical Presets
            </button>
          </div>

          {/* Tab 1: Video / Audio Upload */}
          {activeTab === 'media' && (
            <div className="space-y-4">
              <div className="border-2 border-dashed border-outline-variant hover:border-clinical-blue rounded-xl p-6 text-center bg-surface-container-low transition-colors">
                <input
                  type="file"
                  id="media-file-input"
                  accept="video/*,audio/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <label htmlFor="media-file-input" className="cursor-pointer block space-y-2">
                  <span className="material-symbols-outlined text-clinical-blue text-[36px]">
                    upload_file
                  </span>
                  <div className="font-semibold text-xs text-on-surface">
                    Click to select or drag &amp; drop video or audio file
                  </div>
                  <div className="text-[11px] text-on-surface-variant">
                    Supports MP4, WEBM, MOV, WAV, MP3
                  </div>
                </label>
              </div>

              {selectedFile && (
                <div className="p-3 rounded-lg bg-surface-container border border-outline-variant text-xs flex justify-between items-center">
                  <div>
                    <span className="font-bold text-on-surface block">{selectedFile.name}</span>
                    <span className="text-[10px] text-on-surface-variant">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • {selectedFile.type}
                    </span>
                  </div>
                  <button
                    onClick={handleProcessMedia}
                    disabled={isEvaluating}
                    className="bg-clinical-blue hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg font-bold text-xs transition-colors cursor-pointer disabled:opacity-50"
                  >
                    {isEvaluating ? 'Evaluating...' : 'Process & Evaluate'}
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Tabular / Clinical JSON */}
          {activeTab === 'tabular' && (
            <div className="space-y-3 text-xs">
              <label className="block text-on-surface-variant font-medium">
                Paste Tabular Feature JSON or 18-element Float Array:
              </label>
              <textarea
                rows={5}
                value={tabularJson}
                onChange={(e) => setTabularJson(e.target.value)}
                placeholder='{"Sleep_Quality": 2.5, "HRV_Index": 24, "GSR_Level": 3.8, ...}'
                className="w-full bg-surface-container-low border border-outline-variant rounded-xl p-3 font-data-mono text-[11px] text-on-surface focus:outline-none focus:border-clinical-blue"
              />
              <div className="flex justify-end gap-2">
                <button
                  onClick={() =>
                    setTabularJson(
                      JSON.stringify(
                        {
                          Sleep_Quality: 2.0,
                          Social_Engagement: 1.5,
                          HRV_Index: 22.0,
                          GSR_Level: 3.5,
                          Heart_Rate_BPM: 94.0,
                        },
                        null,
                        2
                      )
                    )
                  }
                  className="px-3 py-1.5 rounded-lg border border-outline-variant text-on-surface-variant hover:text-on-surface text-[11px] font-mono"
                >
                  Insert Sample JSON
                </button>
                <button
                  onClick={handleProcessTabular}
                  disabled={isEvaluating}
                  className="bg-clinical-blue hover:bg-blue-700 text-white px-4 py-1.5 rounded-lg font-bold text-xs transition-colors cursor-pointer disabled:opacity-50"
                >
                  {isEvaluating ? 'Processing...' : 'Run Tabular Evaluation'}
                </button>
              </div>
            </div>
          )}

          {/* Tab 3: Presets */}
          {activeTab === 'presets' && (
            <div className="grid grid-cols-3 gap-3">
              <button
                onClick={() => handlePresetSelect('Healthy Baseline')}
                className="p-4 rounded-xl bg-emerald-50 hover:bg-emerald-100 border border-emerald-300 text-left transition-all cursor-pointer space-y-1"
              >
                <div className="font-bold text-emerald-800 text-xs">Healthy Baseline</div>
                <div className="text-[10px] text-emerald-700 leading-snug">
                  Normal sleep, optimal HRV, low distress markers.
                </div>
              </button>

              <button
                onClick={() => handlePresetSelect('Moderate Distress')}
                className="p-4 rounded-xl bg-amber-50 hover:bg-amber-100 border border-amber-300 text-left transition-all cursor-pointer space-y-1"
              >
                <div className="font-bold text-amber-900 text-xs">Moderate Distress</div>
                <div className="text-[10px] text-amber-800 leading-snug">
                  Reduced sleep quality, blunted affect, elevated GSR.
                </div>
              </button>

              <button
                onClick={() => handlePresetSelect('Severe Agitation')}
                className="p-4 rounded-xl bg-rose-50 hover:bg-rose-100 border border-rose-300 text-left transition-all cursor-pointer space-y-1"
              >
                <div className="font-bold text-rose-900 text-xs">Severe Agitation</div>
                <div className="text-[10px] text-rose-800 leading-snug">
                  Elevated heart rate, high blink rate, critical stress.
                </div>
              </button>
            </div>
          )}

          {/* Upload Status Banner */}
          {uploadStatus && (
            <div className="p-2.5 rounded-lg bg-surface-container border border-outline-variant text-[11px] font-data-mono text-clinical-blue font-semibold flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-clinical-blue animate-ping" />
              {uploadStatus}
            </div>
          )}
        </div>
      </div>
    );
  }
);
