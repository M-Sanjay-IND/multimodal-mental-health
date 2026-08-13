'use client';

import React, { useCallback, useRef, useState } from 'react';
import { SymptomGauges } from './SymptomGauges';
import { FastSHAPChart } from './FastSHAPChart';
import { FacialSaliencyOverlay } from './FacialSaliencyOverlay';
import { ClinicalReportCard } from './ClinicalReportCard';
import { DataUploadModal } from './DataUploadModal';
import { CustomInputIntelligence } from './CustomInputIntelligence';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';
import { useMediaPipeFaceMesh } from '../hooks/useMediaPipeFaceMesh';
import { useAudioProcessor } from '../hooks/useAudioProcessor';

export const Dashboard: React.FC = function Dashboard() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'custom_inputs'>('dashboard');
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState<string | null>(null);

  const diagnosticResults = useDiagnosticResults();
  const {
    connectWebSocket,
    disconnectWebSocket,
    updateModalityStatus,
    injectUploadedPayload,
  } = diagnosticResults;

  const handleRunAnalysis = useCallback(async () => {
    setIsAnalyzing(true);
    setAnalysisStatus('Running DCMF-Net inference & FastSHAP attributions...');
    try {
      await injectUploadedPayload({});
      setAnalysisStatus('DCMF-Net Diagnostic Analysis Complete! Dashboard updated.');
    } catch {
      setAnalysisStatus('Analysis failed. Ensure server is running at localhost:8000.');
    } finally {
      setIsAnalyzing(false);
    }
  }, [injectUploadedPayload]);

  const { startAudio, stopAudio } = useAudioProcessor({
    active: isSessionActive,
  });

  const { startExtraction, stopExtraction } = useMediaPipeFaceMesh({
    videoRef,
    active: isSessionActive,
  });

  const handleSessionToggle = useCallback(async () => {
    setMediaError(null);
    if (!isSessionActive) {
      try {
        setIsSessionActive(true);
        connectWebSocket();

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          const videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 360, frameRate: 30 },
            audio: false,
          });

          if (videoRef.current) {
            videoRef.current.srcObject = videoStream;
            await videoRef.current.play();
          }
        }

        await startExtraction();
        await startAudio();

        updateModalityStatus('visual', 'active');
        updateModalityStatus('acoustic', 'active');
        updateModalityStatus('tabular', 'active');
      } catch (err: any) {
        setMediaError(
          err?.message || 'Camera or Microphone access was denied.'
        );
        updateModalityStatus('visual', 'degraded');
        updateModalityStatus('acoustic', 'degraded');
      }
    } else {
      setIsSessionActive(false);
      stopExtraction();
      stopAudio();

      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }

      disconnectWebSocket();
    }
  }, [
    connectWebSocket,
    disconnectWebSocket,
    isSessionActive,
    startAudio,
    startExtraction,
    stopAudio,
    stopExtraction,
    updateModalityStatus,
  ]);

  const handleVideoFileLoaded = useCallback(
    async (file: File) => {
      setMediaError(null);
      setIsSessionActive(true);
      connectWebSocket();

      if (videoRef.current) {
        if (videoRef.current.srcObject) {
          const stream = videoRef.current.srcObject as MediaStream;
          stream.getTracks().forEach((track) => track.stop());
          videoRef.current.srcObject = null;
        }
        videoRef.current.src = URL.createObjectURL(file);
        videoRef.current.loop = true;
        await videoRef.current.play();
      }

      await startExtraction();
      updateModalityStatus('visual', 'active');
      updateModalityStatus('acoustic', 'active');
      updateModalityStatus('tabular', 'active');
    },
    [connectWebSocket, startExtraction, updateModalityStatus]
  );

  return (
    <div className="min-h-screen bg-background text-on-background w-full pb-12">
      <DataUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onVideoFileLoaded={handleVideoFileLoaded}
      />

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {/* Top Header & Navigation Tabs */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200/80 pb-4">
          <div>
            <h1 className="text-xl font-bold text-on-surface">
              PSYCH-METRIC
            </h1>
            <p className="text-xs text-on-surface-variant mt-0.5">
              Multimodal Psychiatric Assessment & Intelligence Engine
            </p>
          </div>

          {/* Page View Tabs */}
          <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-xl border border-slate-200">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                activeTab === 'dashboard'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Live Assessment Session
            </button>
            <button
              onClick={() => setActiveTab('custom_inputs')}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                activeTab === 'custom_inputs'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Value Inputs Intelligence
            </button>
          </div>

          {/* Action Buttons for Dashboard */}
          {activeTab === 'dashboard' && (
            <div className="flex items-center gap-3">
              <button
                onClick={handleSessionToggle}
                className={`px-4 py-2 rounded-xl font-bold text-xs shadow-sm transition-all cursor-pointer ${
                  isSessionActive
                    ? 'bg-rose-500 hover:bg-rose-600 text-white'
                    : 'bg-emerald-500 hover:bg-emerald-600 text-white'
                }`}
              >
                {isSessionActive ? 'Stop Session' : 'Start Session'}
              </button>

              <button
                onClick={handleRunAnalysis}
                disabled={isAnalyzing}
                className="px-5 py-2 text-white rounded-xl font-bold text-xs shadow-lg transition-all cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
                style={{
                  background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
                  boxShadow: '0 4px 14px rgba(37, 99, 235, 0.35)',
                }}
              >
                {isAnalyzing ? 'Analyzing...' : 'Run DCMF-Net Analysis'}
              </button>

              <button
                onClick={() => setIsUploadModalOpen(true)}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl font-bold text-xs shadow-sm transition-all cursor-pointer border border-slate-200"
              >
                Upload Files
              </button>
            </div>
          )}
        </div>

        {/* Status Banners */}
        {analysisStatus && activeTab === 'dashboard' && (
          <div className="p-3 rounded-xl bg-blue-50 border border-blue-200 text-blue-800 text-xs flex items-center justify-between">
            <span>{analysisStatus}</span>
            <button
              onClick={() => setAnalysisStatus(null)}
              className="text-blue-400 hover:text-blue-600 cursor-pointer ml-4 font-bold"
            >
              ×
            </button>
          </div>
        )}

        {mediaError && activeTab === 'dashboard' && (
          <div className="p-3 rounded-xl bg-amber-50 border border-amber-300 text-amber-900 text-xs flex items-center justify-between">
            <span>{mediaError}</span>
            <button
              onClick={() => setIsUploadModalOpen(true)}
              className="font-bold underline text-clinical-blue cursor-pointer"
            >
              Upload Files Instead
            </button>
          </div>
        )}

        {/* Tab 1: Live Dashboard */}
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-12 gap-widget-gap">
            {/* Video Viewport */}
            <div className="col-span-12 lg:col-span-8 flex flex-col">
              <FacialSaliencyOverlay videoRef={videoRef} isStreaming={isSessionActive} />
            </div>

            {/* Side Panels */}
            <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
              <SymptomGauges />
              <FastSHAPChart />
            </div>

            {/* Clinical Summary */}
            <div className="col-span-12 mt-2">
              <ClinicalReportCard />
            </div>
          </div>
        )}

        {/* Tab 2: Custom Value Inputs Intelligence Page */}
        {activeTab === 'custom_inputs' && <CustomInputIntelligence />}
      </main>
    </div>
  );
};
