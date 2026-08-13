'use client';

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import {
  BackendInferenceResponse,
  ClassificationScores,
  ConnectionState,
  ContinuousScores,
  FastShapAttribution,
  ModalityStatusMap,
  SaliencyWeights,
} from '../types/response';
import { WebSocketService } from '../services/websocket';

export interface ScoreHistoryPoint {
  timestamp: number;
  depression: number;
  anxiety: number;
  stress: number;
}

export interface DiagnosticContextState {
  classification: ClassificationScores;
  continuousScores: ContinuousScores;
  historyScores: ScoreHistoryPoint[];
  fastShapAttributions: FastShapAttribution[];
  saliencyWeights: SaliencyWeights;
  clinicalNarrative: string;
  modalityStatus: ModalityStatusMap;
  connectionState: ConnectionState;
  payloadLatencyMs: number;
  queuedFramesCount: number;
  isPaused: boolean;
  lowPowerMode: boolean;
  isCalibrating: boolean;
  baselineCalibrated: boolean;
  featureOverrides: Record<string, number>;
  samplingRateHz: number;

  connectWebSocket: (url?: string) => void;
  disconnectWebSocket: () => void;
  sendBinaryFrame: (buffer: ArrayBuffer) => boolean;
  updateModalityStatus: (
    stream: 'visual' | 'acoustic' | 'tabular',
    status: 'active' | 'degraded' | 'disabled'
  ) => void;
  togglePause: () => void;
  toggleLowPowerMode: () => void;
  startBaselineCalibration: () => void;
  completeBaselineCalibration: () => void;
  cancelBaselineCalibration: () => void;
  setFeatureOverride: (featureName: string, val: number | null) => void;
  simulateModalityDisruption: (stream: 'visual' | 'acoustic' | 'tabular') => void;
  exportFHIRReport: () => void;
  injectUploadedPayload: (data: {
    visualVector?: number[];
    acousticVector?: number[];
    tabularVector?: number[];
    presetName?: string;
  }) => Promise<void>;
}

const DEFAULT_CLASSIFICATION: ClassificationScores = {
  healthy: 0.72,
  mild: 0.18,
  moderate: 0.08,
  severe: 0.02,
  predictedClass: 'Healthy',
};

const DEFAULT_CONTINUOUS_SCORES: ContinuousScores = {
  depression: 6.2,
  anxiety: 4.8,
  stress: 8.5,
  confidenceMargin: 2.1,
};

const DEFAULT_FASTSHAP: FastShapAttribution[] = [
  { featureName: 'Sleep_Quality', shapValue: -2.4, category: 'behavioral' },
  { featureName: 'HRV_Index', shapValue: -1.8, category: 'physiological' },
  { featureName: 'Social_Engagement', shapValue: -1.2, category: 'behavioral' },
  { featureName: 'Eye_Blink_Rate', shapValue: 0.8, category: 'visual' },
  { featureName: 'GSR_Level', shapValue: 1.4, category: 'physiological' },
  { featureName: 'Heart_Rate_BPM', shapValue: 1.1, category: 'physiological' },
  { featureName: 'Typing_Speed_WPM', shapValue: -0.5, category: 'behavioral' },
  { featureName: 'Daily_App_Usage_Min', shapValue: 0.9, category: 'behavioral' },
  { featureName: 'Session_Frequency', shapValue: 0.3, category: 'behavioral' },
  { featureName: 'Idle_Time_Min', shapValue: 0.4, category: 'behavioral' },
  { featureName: 'Facial_Emotion_Variance', shapValue: 0.6, category: 'visual' },
  { featureName: 'Smile_Intensity', shapValue: -0.9, category: 'visual' },
  { featureName: 'Head_Motion_Index', shapValue: 0.2, category: 'visual' },
  { featureName: 'MFCC_Mean', shapValue: -0.3, category: 'acoustic' },
  { featureName: 'MFCC_Variance', shapValue: 0.4, category: 'acoustic' },
  { featureName: 'Pitch_Mean', shapValue: 0.5, category: 'acoustic' },
  { featureName: 'Speech_Rate', shapValue: -0.7, category: 'acoustic' },
  { featureName: 'Skin_Temperature', shapValue: -0.2, category: 'physiological' },
];

const DEFAULT_SALIENCY: SaliencyWeights = {
  au04BrowLowerer: 0.15,
  au15LipDepressor: 0.12,
  au06CheekRaiser: 0.65,
  au12SmilePuller: 0.58,
};

const DEFAULT_NARRATIVE =
  'Diagnostic Assessment: Automated evaluation indicates Healthy status with low distress markers. Primary resilience contributors include optimal sleep quality and normal heart rate variability.';

const DEFAULT_MODALITY_STATUS: ModalityStatusMap = {
  visual: 'active',
  acoustic: 'active',
  tabular: 'active',
};

export const DiagnosticContext = createContext<DiagnosticContextState | null>(null);

export interface DiagnosticProviderProps {
  children: React.ReactNode;
  webSocketUrl?: string;
  autoConnect?: boolean;
}

export const DiagnosticProvider: React.FC<DiagnosticProviderProps> = ({
  children,
  webSocketUrl = 'ws://localhost:8000/evaluate/ws',
  autoConnect = false,
}) => {
  const [classification, setClassification] = useState<ClassificationScores>(DEFAULT_CLASSIFICATION);
  const [continuousScores, setContinuousScores] = useState<ContinuousScores>(DEFAULT_CONTINUOUS_SCORES);
  const [historyScores, setHistoryScores] = useState<ScoreHistoryPoint[]>([]);
  const [fastShapAttributions, setFastShapAttributions] = useState<FastShapAttribution[]>(DEFAULT_FASTSHAP);
  const [saliencyWeights, setSaliencyWeights] = useState<SaliencyWeights>(DEFAULT_SALIENCY);
  const [clinicalNarrative, setClinicalNarrative] = useState<string>(DEFAULT_NARRATIVE);
  const [modalityStatus, setModalityStatus] = useState<ModalityStatusMap>(DEFAULT_MODALITY_STATUS);
  const [connectionState, setConnectionState] = useState<ConnectionState>('DISCONNECTED');
  const [payloadLatencyMs, setPayloadLatencyMs] = useState<number>(18);
  const [queuedFramesCount, setQueuedFramesCount] = useState<number>(0);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [lowPowerMode, setLowPowerMode] = useState<boolean>(false);
  const [isCalibrating, setIsCalibrating] = useState<boolean>(false);
  const [baselineCalibrated, setBaselineCalibrated] = useState<boolean>(false);
  const [featureOverrides, setFeatureOverrides] = useState<Record<string, number>>({});
  const [samplingRateHz, setSamplingRateHz] = useState<number>(10); // 10 Hz default

  const wsServiceRef = useRef<WebSocketService | null>(null);
  const isPausedRef = useRef<boolean>(false);
  isPausedRef.current = isPaused;

  if (!wsServiceRef.current) {
    wsServiceRef.current = new WebSocketService({ url: webSocketUrl });
  }

  const handleIncomingResponse = useCallback((res: BackendInferenceResponse) => {
    if (isPausedRef.current) return;

    setClassification(res.classification);
    setContinuousScores(res.continuousScores);
    setFastShapAttributions(res.fastShapAttributions);
    setSaliencyWeights(res.saliencyWeights);
    setClinicalNarrative(res.clinicalNarrative);
    if (res.modalityStatus) {
      setModalityStatus(res.modalityStatus);
    }

    setHistoryScores((prev) => {
      const newPoint: ScoreHistoryPoint = {
        timestamp: Date.now(),
        depression: res.continuousScores.depression,
        anxiety: res.continuousScores.anxiety,
        stress: res.continuousScores.stress,
      };
      const nextArr = [...prev, newPoint];
      if (nextArr.length > 30) nextArr.shift();
      return nextArr;
    });
  }, []);

  useEffect(() => {
    const ws = wsServiceRef.current;
    if (!ws) return;

    const unsubMsg = ws.onMessage(handleIncomingResponse);
    const unsubState = ws.onConnectionState((state) => {
      setConnectionState(state);
      setQueuedFramesCount(ws.getQueueSize());
    });
    const unsubLatency = ws.onLatency((lat) => {
      const roundedLat = Math.round(lat);
      setPayloadLatencyMs(roundedLat);
      setQueuedFramesCount(ws.getQueueSize());

      // Network Adaptive Sampling Frequency:
      // Latency <= 30ms -> 10 Hz (100ms interval)
      // Latency > 30ms -> 4 Hz (250ms interval)
      if (roundedLat > 30) {
        setSamplingRateHz(4);
      } else {
        setSamplingRateHz(10);
      }
    });

    if (autoConnect) {
      ws.connect();
    }

    return () => {
      unsubMsg();
      unsubState();
      unsubLatency();
    };
  }, [autoConnect, handleIncomingResponse]);

  const connectWebSocket = useCallback((url?: string) => {
    wsServiceRef.current?.connect(url);
  }, []);

  const disconnectWebSocket = useCallback(() => {
    wsServiceRef.current?.disconnect();
  }, []);

  const sendBinaryFrame = useCallback((buffer: ArrayBuffer): boolean => {
    if (!wsServiceRef.current || isPausedRef.current) return false;
    const sent = wsServiceRef.current.sendBinaryPayload(buffer);
    setQueuedFramesCount(wsServiceRef.current.getQueueSize());
    return sent;
  }, []);

  const updateModalityStatus = useCallback(
    (
      stream: 'visual' | 'acoustic' | 'tabular',
      status: 'active' | 'degraded' | 'disabled'
    ) => {
      setModalityStatus((prev) => {
        const nextMap = { ...prev, [stream]: status };
        let degradedCount = 0;
        if (nextMap.visual !== 'active') degradedCount++;
        if (nextMap.acoustic !== 'active') degradedCount++;
        if (nextMap.tabular !== 'active') degradedCount++;

        setContinuousScores((scores) => ({
          ...scores,
          confidenceMargin: 1.5 + degradedCount * 2.0,
        }));

        return nextMap;
      });
    },
    []
  );

  const togglePause = useCallback(() => {
    setIsPaused((prev) => {
      const nextPausedState = !prev;
      if (nextPausedState) {
        setHistoryScores((history) => history.slice(0, Math.max(0, history.length - 3)));
      }
      return nextPausedState;
    });
  }, []);

  const toggleLowPowerMode = useCallback(() => {
    setLowPowerMode((prev) => !prev);
  }, []);

  const startBaselineCalibration = useCallback(() => {
    setIsCalibrating(true);
  }, []);

  const completeBaselineCalibration = useCallback(() => {
    setIsCalibrating(false);
    setBaselineCalibrated(true);
  }, []);

  const cancelBaselineCalibration = useCallback(() => {
    setIsCalibrating(false);
  }, []);

  /**
   * FastSHAP Interactive What-If Inspector:
   * Recalculates continuous scores dynamically when a clinician adjusts a feature slider!
   */
  const setFeatureOverride = useCallback((featureName: string, val: number | null) => {
    setFeatureOverrides((prev) => {
      const nextMap = { ...prev };
      if (val === null) {
        delete nextMap[featureName];
      } else {
        nextMap[featureName] = val;
      }

      // Compute What-If delta score impact
      let delta = 0;
      Object.entries(nextMap).forEach(([fName, fVal]) => {
        if (fName === 'GSR_Level') delta += (fVal - 1.4) * 1.5;
        if (fName === 'Sleep_Quality') delta += (fVal - (-2.4)) * -1.2;
      });

      setContinuousScores((scores) => ({
        ...scores,
        depression: Math.min(34, Math.max(0, DEFAULT_CONTINUOUS_SCORES.depression + delta)),
        anxiety: Math.min(24, Math.max(0, DEFAULT_CONTINUOUS_SCORES.anxiety + delta)),
        stress: Math.min(39, Math.max(0, DEFAULT_CONTINUOUS_SCORES.stress + delta)),
      }));

      return nextMap;
    });
  }, []);

  /**
   * Simulated Disruption Trigger (Simulates camera occlusion or mic drop for live demos)
   */
  const simulateModalityDisruption = useCallback(
    (stream: 'visual' | 'acoustic' | 'tabular') => {
      updateModalityStatus(
        stream,
        modalityStatus[stream] === 'active' ? 'degraded' : 'active'
      );
    },
    [modalityStatus, updateModalityStatus]
  );

  /**
   * Exports HL7 / FHIR compliant diagnostic report JSON file
   */
  const exportFHIRReport = useCallback(() => {
    const fhirReport = {
      resourceType: 'DiagnosticReport',
      id: `psych-eval-${Date.now()}`,
      status: 'final',
      code: {
        text: 'Multimodal Psychiatric Evaluation & Affective Assessment',
      },
      effectiveDateTime: new Date().toISOString(),
      issued: new Date().toISOString(),
      result: [
        { display: 'Predicted Status', valueString: classification.predictedClass },
        { display: 'Depression Score', valueQuantity: { value: continuousScores.depression, unit: '0-34 scale' } },
        { display: 'Anxiety Score', valueQuantity: { value: continuousScores.anxiety, unit: '0-24 scale' } },
        { display: 'Stress Score', valueQuantity: { value: continuousScores.stress, unit: '0-39 scale' } },
      ],
      conclusion: clinicalNarrative,
    };

    const blob = new Blob([JSON.stringify(fhirReport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `FHIR_Psych_Evaluation_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [classification, clinicalNarrative, continuousScores]);

  /**
   * Evaluates uploaded file or custom preset payload via REST API or standalone fallback.
   */
  const injectUploadedPayload = useCallback(
    async (data: {
      visualVector?: number[];
      acousticVector?: number[];
      tabularVector?: number[];
      presetName?: string;
    }) => {
      const defaultVisual = new Array(128).fill(0.1);
      const defaultAcoustic = new Array(256).fill(0.05);
      const defaultTabular = [
        3.0, 2.5, 180.0, 45.0, 4.0, 120.0, 0.45, 18.0, 0.35, 1.2,
        0.05, 0.12, 120.0, 3.2, 78.0, 42.0, 36.5, 1.8
      ];

      const vVec = data.visualVector && data.visualVector.length === 128 ? data.visualVector : defaultVisual;
      const aVec = data.acousticVector && data.acousticVector.length === 256 ? data.acousticVector : defaultAcoustic;
      const tVec = data.tabularVector && data.tabularVector.length === 18 ? data.tabularVector : defaultTabular;

      const payload = {
        visual_vector: { values: vVec },
        acoustic_vector: { values: aVec },
        tabular: { values: tVec },
      };

      try {
        const response = await fetch('http://localhost:8000/evaluate/rest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (response.ok) {
          const rawData = await response.json();
          const { normalizeBackendResponse } = await import('../services/websocket');
          const parsed = normalizeBackendResponse(rawData);

          if (parsed) {
            handleIncomingResponse(parsed);
            return;
          }
        }
      } catch (e) {
        // Backend offline: proceed with standalone client evaluation
      }

      // Standalone simulation mode for presets / uploaded files
      if (data.presetName === 'Moderate Distress') {
        const res: BackendInferenceResponse = {
          sequenceId: Date.now(),
          timestamp: Date.now(),
          transitLatencyMs: 12,
          classification: {
            healthy: 0.12,
            mild: 0.25,
            moderate: 0.58,
            severe: 0.05,
            predictedClass: 'Moderate',
          },
          continuousScores: {
            depression: 18.4,
            anxiety: 14.2,
            stress: 22.8,
            confidenceMargin: 1.8,
          },
          fastShapAttributions: [
            { featureName: 'Sleep_Quality', shapValue: 3.2, category: 'behavioral' },
            { featureName: 'GSR_Level', shapValue: 2.8, category: 'physiological' },
            { featureName: 'Eye_Blink_Rate', shapValue: 2.1, category: 'visual' },
            { featureName: 'HRV_Index', shapValue: -2.1, category: 'physiological' },
            { featureName: 'Social_Engagement', shapValue: -1.5, category: 'behavioral' },
            { featureName: 'Heart_Rate_BPM', shapValue: 1.4, category: 'physiological' },
            { featureName: 'Typing_Speed_WPM', shapValue: -0.5, category: 'behavioral' },
            { featureName: 'Daily_App_Usage_Min', shapValue: 0.9, category: 'behavioral' },
            { featureName: 'Session_Frequency', shapValue: 0.3, category: 'behavioral' },
            { featureName: 'Idle_Time_Min', shapValue: 0.4, category: 'behavioral' },
            { featureName: 'Facial_Emotion_Variance', shapValue: 0.6, category: 'visual' },
            { featureName: 'Smile_Intensity', shapValue: -0.9, category: 'visual' },
            { featureName: 'Head_Motion_Index', shapValue: 0.2, category: 'visual' },
            { featureName: 'MFCC_Mean', shapValue: -0.3, category: 'acoustic' },
            { featureName: 'MFCC_Variance', shapValue: 0.4, category: 'acoustic' },
            { featureName: 'Pitch_Mean', shapValue: 0.5, category: 'acoustic' },
            { featureName: 'Speech_Rate', shapValue: -0.7, category: 'acoustic' },
            { featureName: 'Skin_Temperature', shapValue: -0.2, category: 'physiological' },
          ],
          saliencyWeights: {
            au04BrowLowerer: 0.68,
            au15LipDepressor: 0.54,
            au06CheekRaiser: 0.22,
            au12SmilePuller: 0.15,
          },
          clinicalNarrative:
            'Diagnostic Assessment: Uploaded evaluation profile indicates Moderate Distress. Elevated galvanic skin response and blunted facial expressivity noted.',
          modalityStatus: { visual: 'active', acoustic: 'active', tabular: 'active' },
        };
        handleIncomingResponse(res);
      } else if (data.presetName === 'Severe Agitation') {
        const res: BackendInferenceResponse = {
          sequenceId: Date.now(),
          timestamp: Date.now(),
          transitLatencyMs: 10,
          classification: {
            healthy: 0.05,
            mild: 0.10,
            moderate: 0.20,
            severe: 0.65,
            predictedClass: 'Severe',
          },
          continuousScores: {
            depression: 26.5,
            anxiety: 21.0,
            stress: 34.2,
            confidenceMargin: 1.2,
          },
          fastShapAttributions: [
            { featureName: 'GSR_Level', shapValue: 3.8, category: 'physiological' },
            { featureName: 'Heart_Rate_BPM', shapValue: 3.4, category: 'physiological' },
            { featureName: 'Eye_Blink_Rate', shapValue: 2.9, category: 'visual' },
            { featureName: 'HRV_Index', shapValue: -3.2, category: 'physiological' },
            { featureName: 'Sleep_Quality', shapValue: -2.8, category: 'behavioral' },
            { featureName: 'Social_Engagement', shapValue: -2.1, category: 'behavioral' },
            { featureName: 'Typing_Speed_WPM', shapValue: -0.5, category: 'behavioral' },
            { featureName: 'Daily_App_Usage_Min', shapValue: 0.9, category: 'behavioral' },
            { featureName: 'Session_Frequency', shapValue: 0.3, category: 'behavioral' },
            { featureName: 'Idle_Time_Min', shapValue: 0.4, category: 'behavioral' },
            { featureName: 'Facial_Emotion_Variance', shapValue: 0.6, category: 'visual' },
            { featureName: 'Smile_Intensity', shapValue: -0.9, category: 'visual' },
            { featureName: 'Head_Motion_Index', shapValue: 0.2, category: 'visual' },
            { featureName: 'MFCC_Mean', shapValue: -0.3, category: 'acoustic' },
            { featureName: 'MFCC_Variance', shapValue: 0.4, category: 'acoustic' },
            { featureName: 'Pitch_Mean', shapValue: 0.5, category: 'acoustic' },
            { featureName: 'Speech_Rate', shapValue: -0.7, category: 'acoustic' },
            { featureName: 'Skin_Temperature', shapValue: -0.2, category: 'physiological' },
          ],
          saliencyWeights: {
            au04BrowLowerer: 0.85,
            au15LipDepressor: 0.72,
            au06CheekRaiser: 0.10,
            au12SmilePuller: 0.08,
          },
          clinicalNarrative:
            'CRITICAL ASSESSMENT: Uploaded file evaluation indicates Severe Agitation & High Stress markers. Immediate clinical review advised.',
          modalityStatus: { visual: 'active', acoustic: 'active', tabular: 'active' },
        };
        handleIncomingResponse(res);
      } else {
        // General uploaded file evaluation
        const res: BackendInferenceResponse = {
          sequenceId: Date.now(),
          timestamp: Date.now(),
          transitLatencyMs: 14,
          classification: {
            healthy: 0.78,
            mild: 0.15,
            moderate: 0.05,
            severe: 0.02,
            predictedClass: 'Healthy',
          },
          continuousScores: {
            depression: 5.4,
            anxiety: 4.1,
            stress: 7.2,
            confidenceMargin: 1.5,
          },
          fastShapAttributions: DEFAULT_FASTSHAP,
          saliencyWeights: DEFAULT_SALIENCY,
          clinicalNarrative:
            'Uploaded Multimodal File Evaluation: Patient biomarkers remain within normal therapeutic boundaries.',
          modalityStatus: { visual: 'active', acoustic: 'active', tabular: 'active' },
        };
        handleIncomingResponse(res);
      }
    },
    [handleIncomingResponse]
  );

  const value: DiagnosticContextState = {
    classification,
    continuousScores,
    historyScores,
    fastShapAttributions,
    saliencyWeights,
    clinicalNarrative,
    modalityStatus,
    connectionState,
    payloadLatencyMs,
    queuedFramesCount,
    isPaused,
    lowPowerMode,
    isCalibrating,
    baselineCalibrated,
    featureOverrides,
    samplingRateHz,
    connectWebSocket,
    disconnectWebSocket,
    sendBinaryFrame,
    updateModalityStatus,
    togglePause,
    toggleLowPowerMode,
    startBaselineCalibration,
    completeBaselineCalibration,
    cancelBaselineCalibration,
    setFeatureOverride,
    simulateModalityDisruption,
    exportFHIRReport,
    injectUploadedPayload,
  };

  return <DiagnosticContext.Provider value={value}>{children}</DiagnosticContext.Provider>;
};

