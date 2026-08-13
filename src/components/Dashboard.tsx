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
    setAnalysisStatus('Executing DCMF-Net model inference & Shapley attributions...');
    try {
      await injectUploadedPayload({});
      setAnalysisStatus('DCMF-Net Model Inference Complete.');
    } catch {
      setAnalysisStatus('Analysis error. Ensure model server is running at localhost:8000.');
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
    <div className="min-h-screen bg-[#09090b] text-[#fafafa] w-full pb-12 font-sans">
      <DataUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onVideoFileLoaded={handleVideoFileLoaded}
      />

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {/* Header Navigation */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white font-mono">
              PSYCH-METRIC // MULTIMODAL AI
            </h1>
            <p className="text-xs text-zinc-400 mt-0.5 font-mono">
              Affective Assessment & Multi-Task Diagnostic System
            </p>
          </div>

          {/* Navigation View Switcher */}
          <div className="flex items-center gap-1 bg-zinc-900 p-1 rounded-lg border border-zinc-800 font-mono text-xs">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`px-4 py-1.5 rounded text-xs font-semibold transition-all cursor-pointer ${
                activeTab === 'dashboard'
                  ? 'bg-white text-black font-bold shadow'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              Live Assessment
            </button>
            <button
              onClick={() => setActiveTab('custom_inputs')}
              className={`px-4 py-1.5 rounded text-xs font-semibold transition-all cursor-pointer ${
                activeTab === 'custom_inputs'
                  ? 'bg-white text-black font-bold shadow'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              Model Value Inputs
            </button>
          </div>

          {/* Action Buttons for Dashboard */}
          {activeTab === 'dashboard' && (
            <div className="flex items-center gap-3 font-mono text-xs">
              <button
                onClick={handleSessionToggle}
                className={`px-4 py-2 rounded-lg font-bold transition-all cursor-pointer ${
                  isSessionActive
                    ? 'bg-zinc-100 text-black hover:bg-white'
                    : 'bg-zinc-900 hover:bg-zinc-800 text-white border border-zinc-700'
                }`}
              >
                {isSessionActive ? 'Stop Stream' : 'Start Camera/Mic'}
              </button>

              <button
                onClick={handleRunAnalysis}
                disabled={isAnalyzing}
                className="px-5 py-2 bg-white hover:bg-zinc-200 text-black rounded-lg font-bold transition-all cursor-pointer disabled:opacity-50"
              >
                {isAnalyzing ? 'Analyzing...' : 'Run DCMF-Net Inference'}
              </button>

              <button
                onClick={() => setIsUploadModalOpen(true)}
                className="px-3 py-2 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 rounded-lg border border-zinc-800 transition-all cursor-pointer"
              >
                Upload File
              </button>
            </div>
          )}
        </div>

        {/* Banners */}
        {analysisStatus && activeTab === 'dashboard' && (
          <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-700 text-zinc-200 text-xs font-mono flex items-center justify-between">
            <span>{analysisStatus}</span>
            <button
              onClick={() => setAnalysisStatus(null)}
              className="text-zinc-500 hover:text-white cursor-pointer ml-4 font-bold"
            >
              ×
            </button>
          </div>
        )}

        {mediaError && activeTab === 'dashboard' && (
          <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-700 text-zinc-300 text-xs font-mono flex items-center justify-between">
            <span>{mediaError}</span>
            <button
              onClick={() => setIsUploadModalOpen(true)}
              className="font-bold underline text-white cursor-pointer"
            >
              Upload Media Instead
            </button>
          </div>
        )}

        {/* Live Dashboard View */}
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-8 flex flex-col">
              <FacialSaliencyOverlay videoRef={videoRef} isStreaming={isSessionActive} />
            </div>

            <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
              <SymptomGauges />
              <FastSHAPChart />
            </div>

            <div className="col-span-12">
              <ClinicalReportCard />
            </div>
          </div>
        )}

        {/* Custom Model Value Inputs View */}
        {activeTab === 'custom_inputs' && <CustomInputIntelligence />}
      </main>
    </div>
  );
};
