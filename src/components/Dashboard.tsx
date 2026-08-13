'use client';

import React, { useCallback, useRef, useState } from 'react';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { SymptomGauges } from './SymptomGauges';
import { FastSHAPChart } from './FastSHAPChart';
import { FacialSaliencyOverlay } from './FacialSaliencyOverlay';
import { ClinicalReportCard } from './ClinicalReportCard';
import { BaselineCalibrationModal } from './BaselineCalibrationModal';
import { useDiagnosticResults } from '../hooks/useDiagnosticResults';
import { useMediaPipeFaceMesh } from '../hooks/useMediaPipeFaceMesh';
import { useAudioProcessor } from '../hooks/useAudioProcessor';

export const Dashboard: React.FC = function Dashboard() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isSessionActive, setIsSessionActive] = useState(false);

  const {
    connectWebSocket,
    disconnectWebSocket,
    updateModalityStatus,
    isCalibrating,
    completeBaselineCalibration,
    cancelBaselineCalibration,
  } = useDiagnosticResults();

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
    if (!isSessionActive) {
      try {
        setIsSessionActive(true);
        connectWebSocket();

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          const videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 360, frameRate: 30 },
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
      } catch {
        // Exception handling for camera/mic permissions
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

  return (
    <div className="flex h-screen overflow-hidden bg-background text-on-background w-full">
      {/* 15-Second Baseline Calibration Modal */}
      <BaselineCalibrationModal
        isOpen={isCalibrating}
        onComplete={completeBaselineCalibration}
        onCancel={cancelBaselineCalibration}
      />

      {/* SideNavBar Navigation */}
      <Sidebar onSessionToggle={handleSessionToggle} isSessionActive={isSessionActive} />

      {/* Main Content Area */}
      <div className="flex-1 ml-0 md:ml-nav-width flex flex-col h-full bg-background relative overflow-hidden">
        {/* TopNavBar Header */}
        <Header onSessionToggle={handleSessionToggle} isSessionActive={isSessionActive} />

        {/* Dashboard Content */}
        <main className="flex-1 mt-top-bar-height p-container-padding overflow-y-auto w-full">
          <div className="max-w-7xl mx-auto space-y-6">
            {/* Session Information Header */}
            <div className="flex items-center justify-between mb-4">
              <div>
                <h1 className="font-section-header text-section-header text-on-surface">
                  Session 42A - Active Monitoring
                </h1>
                <p className="font-body-md text-body-md text-on-surface-variant">
                  Patient ID: 884-291-B
                </p>
              </div>
            </div>

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

