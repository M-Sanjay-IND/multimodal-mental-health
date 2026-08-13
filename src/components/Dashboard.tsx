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
   * Toggle Session Trigger.
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
        // Exception handling
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
    <div className="bg-[#0A0A0C] text-[#e2e1eb] h-screen overflow-hidden flex flex-col antialiased font-sans">
      {/* 15-Second Baseline Calibration Modal */}
      <BaselineCalibrationModal
        isOpen={isCalibrating}
        onComplete={completeBaselineCalibration}
        onCancel={cancelBaselineCalibration}
      />

      {/* Telemetry Header */}
      <Header onSessionToggle={handleSessionToggle} isSessionActive={isSessionActive} />

      {/* Main Container */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Clinical Sidebar */}
        <Sidebar onSessionToggle={handleSessionToggle} isSessionActive={isSessionActive} />

        {/* Main Dashboard Grid */}
        <main className="flex-1 overflow-y-auto bg-[#0A0A0C] p-4 flex flex-col gap-4">
          {/* Top Grid: Camera Viewport (8 cols) & FastSHAP Panel (4 cols) */}
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-12 lg:col-span-8">
              <FacialSaliencyOverlay videoRef={videoRef} isStreaming={isSessionActive} />
            </div>
            <div className="col-span-12 lg:col-span-4">
              <FastSHAPChart />
            </div>
          </div>

          {/* Bottom Grid: Symptom Meters (6 cols) & Clinical Summary (6 cols) */}
          <div className="grid grid-cols-12 gap-4 flex-1">
            <div className="col-span-12 lg:col-span-6">
              <SymptomGauges />
            </div>
            <div className="col-span-12 lg:col-span-6">
              <ClinicalReportCard />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};
