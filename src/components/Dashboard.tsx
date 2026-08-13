'use client';

import React, { useCallback, useRef, useState } from 'react';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { SymptomGauges } from './SymptomGauges';
import { FastSHAPChart } from './FastSHAPChart';
import { FacialSaliencyOverlay } from './FacialSaliencyOverlay';
import { ClinicalReportCard } from './ClinicalReportCard';
import { BaselineCalibrationModal } from './BaselineCalibrationModal';
import { DataUploadModal } from './DataUploadModal';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';
import { useMediaPipeFaceMesh } from '../hooks/useMediaPipeFaceMesh';
import { useAudioProcessor } from '../hooks/useAudioProcessor';

export const Dashboard: React.FC = function Dashboard() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
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
    isCalibrating,
    completeBaselineCalibration,
    cancelBaselineCalibration,
    injectUploadedPayload,
  } = diagnosticResults;

  const handleRunAnalysis = useCallback(async () => {
    setIsAnalyzing(true);
    setAnalysisStatus('⚡ Running DCMF-Net inference & FastSHAP attributions...');
    try {
      await injectUploadedPayload({});
      setAnalysisStatus('✅ DCMF-Net Diagnostic Analysis Complete! Dashboard updated.');
    } catch {
      setAnalysisStatus('⚠️ Analysis failed. Ensure server is running at localhost:8000.');
    } finally {
      setIsAnalyzing(false);
    }
  }, [injectUploadedPayload]);

  // Audio Processing Hook
  const { startAudio, stopAudio } = useAudioProcessor({
    active: isSessionActive,
  });

  // MediaPipe Face Mesh Hook
  const { startExtraction, stopExtraction } = useMediaPipeFaceMesh({
    videoRef,
    active: isSessionActive,
  });

  /**
   * Toggle Session Stream Trigger.
   */
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
          err?.message || 'Camera or Microphone access was denied. You can also upload media files or tabular data inputs.'
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
    <div className="flex h-screen overflow-hidden bg-background text-on-background w-full">
      {/* 15-Second Baseline Calibration Modal */}
      <BaselineCalibrationModal
        isOpen={isCalibrating}
        onComplete={completeBaselineCalibration}
        onCancel={cancelBaselineCalibration}
      />

      {/* Data Upload & Multimodal Input Modal */}
      <DataUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onVideoFileLoaded={handleVideoFileLoaded}
      />

      {/* SideNavBar Navigation */}
      <Sidebar onSessionToggle={handleSessionToggle} isSessionActive={isSessionActive} />

      {/* Main Content Area */}
      <div className="flex-1 ml-0 md:ml-nav-width flex flex-col h-full bg-background relative overflow-hidden">
        {/* TopNavBar Header */}
        <Header
          onSessionToggle={handleSessionToggle}
          isSessionActive={isSessionActive}
          onUploadClick={() => setIsUploadModalOpen(true)}
        />

        {/* Dashboard Content */}
        <main className="flex-1 mt-top-bar-height p-container-padding overflow-y-auto w-full">
          <div className="max-w-7xl mx-auto space-y-6">
            {/* Session Information Header */}
            <div className="flex items-center justify-between mb-4">
              <div>
                <h1 className="font-section-header text-section-header text-on-surface">
                  Session 42A - Multimodal Assessment
                </h1>
                <p className="font-body-md text-body-md text-on-surface-variant">
                  Patient ID: 884-291-B • Real-time Diagnostic Pipeline
                </p>
              </div>

              <div className="flex items-center gap-3">
                {/* Prominent Analyze Button */}
                <button
                  onClick={handleRunAnalysis}
                  disabled={isAnalyzing}
                  className="px-5 py-2.5 text-white rounded-xl font-bold text-sm shadow-lg transition-all cursor-pointer flex items-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
                  style={{
                    background: 'linear-gradient(135deg, #2563eb, #1d4ed8)',
                    boxShadow: '0 4px 14px rgba(37, 99, 235, 0.35)',
                  }}
                >
                  <span className="material-symbols-outlined text-[18px]">neurology</span>
                  {isAnalyzing ? 'Analyzing with DCMF-Net...' : '⚡ Run DCMF-Net Diagnostic Model'}
                </button>

                <button
                  onClick={() => setIsUploadModalOpen(true)}
                  className="px-4 py-2 bg-clinical-blue hover:bg-blue-700 text-white rounded-xl font-bold text-xs shadow-sm transition-all cursor-pointer flex items-center gap-2"
                >
                  <span className="material-symbols-outlined text-[18px]">upload</span>
                  Upload Files
                </button>
              </div>
            </div>

            {/* Analysis Status Banner */}
            {analysisStatus && (
              <div className="p-3 rounded-xl bg-blue-50 border border-blue-200 text-blue-800 text-xs mb-4 flex items-center justify-between">
                <span>{analysisStatus}</span>
                <button
                  onClick={() => setAnalysisStatus(null)}
                  className="text-blue-400 hover:text-blue-600 cursor-pointer ml-4 font-bold"
                >
                  ×
                </button>
              </div>
            )}

            {/* Media Permission Warning Banner if blocked */}
            {mediaError && (
              <div className="p-4 rounded-xl bg-amber-50 border border-amber-300 text-amber-900 text-xs flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-[20px] text-amber-600">warning</span>
                  <span>{mediaError}</span>
                </div>
                <button
                  onClick={() => setIsUploadModalOpen(true)}
                  className="font-bold underline text-clinical-blue cursor-pointer"
                >
                  Upload Files Instead
                </button>
              </div>
            )}

            {/* Bento Grid Layout */}
            <div className="grid grid-cols-12 gap-widget-gap">
              {/* Main Video Viewport (Col 8) */}
              <div className="col-span-12 lg:col-span-8 flex flex-col">
                <FacialSaliencyOverlay videoRef={videoRef} isStreaming={isSessionActive} />
              </div>

              {/* Side Panels (Col 4) */}
              <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
                <SymptomGauges />
                <FastSHAPChart />
              </div>

              {/* Clinical Summary Bottom Row (Col 12) */}
              <div className="col-span-12 mt-2">
                <ClinicalReportCard />
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};


