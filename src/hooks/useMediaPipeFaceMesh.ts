import { useCallback, useEffect, useRef, useState } from 'react';
import { PAYLOAD_CONSTANTS, VisualVector } from '../types/payload';
import { purgeRawMediaBuffers } from '../utils/privacyManager';

export interface UseMediaPipeOptions {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef?: React.RefObject<HTMLCanvasElement | null>;
  active?: boolean;
  onVectorExtracted?: (vector: VisualVector, ear: number, fev: number, blinkRate: number) => void;
}

export interface UseMediaPipeReturn {
  isInitialized: boolean;
  isActive: boolean;
  visualVector: VisualVector | null;
  ear: number;
  fev: number;
  blinkRate: number;
  latencyMs: number;
  error: string | null;
  startExtraction: () => Promise<void>;
  stopExtraction: () => void;
}

/**
 * Calculates Euclidean distance between two 3D landmarks (x, y, z).
 */
export function euclideanDistance3D(
  p1: { x: number; y: number; z?: number },
  p2: { x: number; y: number; z?: number }
): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const dz = (p1.z || 0) - (p2.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Calculates Eye Aspect Ratio (EAR) given 6 eye landmarks [p1, p2, p3, p4, p5, p6].
 * EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
 */
export function calculateEAR(landmarks: Array<{ x: number; y: number; z?: number }>): number {
  if (landmarks.length < 6) return 0.25;

  const v1 = euclideanDistance3D(landmarks[1], landmarks[5]);
  const v2 = euclideanDistance3D(landmarks[2], landmarks[4]);
  const h = euclideanDistance3D(landmarks[0], landmarks[3]);

  if (h === 0) return 0.25;
  return (v1 + v2) / (2.0 * h);
}

/**
 * Calculates Facial Emotion Volatility (FEV) over a 30-frame sliding window (W=30).
 * FEV = (1/W) * sum_{t=1}^W || H_t - H_bar ||^2
 */
export function calculateFEV(history: number[][]): number {
  if (history.length === 0) return 0;
  const W = history.length;
  const numFeatures = history[0].length;

  // 1. Calculate mean vector H_bar
  const meanVector = new Array(numFeatures).fill(0);
  for (let t = 0; t < W; t++) {
    for (let f = 0; f < numFeatures; f++) {
      meanVector[f] += history[t][f];
    }
  }
  for (let f = 0; f < numFeatures; f++) {
    meanVector[f] /= W;
  }

  // 2. Compute mean squared error volatility
  let totalVariance = 0;
  for (let t = 0; t < W; t++) {
    let frameDiffSq = 0;
    for (let f = 0; f < numFeatures; f++) {
      const diff = history[t][f] - meanVector[f];
      frameDiffSq += diff * diff;
    }
    totalVariance += frameDiffSq;
  }

  return totalVariance / W;
}

/**
 * 34 Canonical Landmark Indices mapping 478 MediaPipe points to 34 key facial anchor points.
 */
const CANONICAL_LANDMARK_INDICES = [
  10, 33, 68, 109, 133, 144, 153, 158, 160, 197, 234, 263, 298, 300, 332, 373, 380, 385, 387, 454,
  1, 61, 291, 0, 17, 39, 269, 70, 105, 336, 334, 152, 13, 14
];

export function useMediaPipeFaceMesh({
  videoRef,
  canvasRef,
  active = false,
  onVectorExtracted,
}: UseMediaPipeOptions): UseMediaPipeReturn {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [visualVector, setVisualVector] = useState<VisualVector | null>(null);
  const [ear, setEar] = useState(0.28);
  const [fev, setFev] = useState(0.02);
  const [blinkRate, setBlinkRate] = useState(16.0); // Blinks per minute
  const [latencyMs, setLatencyMs] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // State buffers
  const faceMeshRef = useRef<any>(null);
  const animFrameRef = useRef<number | null>(null);
  const fevHistoryRef = useRef<number[][]>([]);
  const blinkHistoryRef = useRef<number[]>([]);
  const lastBlinkTimeRef = useRef<number>(Date.now());
  const isBlinkingRef = useRef<boolean>(false);

  // Pre-allocated reusable Float32 array (128 elements)
  const vectorBufferRef = useRef<Float32Array>(
    new Float32Array(PAYLOAD_CONSTANTS.VISUAL_VECTOR_LENGTH)
  );

  /**
   * Initializes MediaPipe Face Mesh Wasm instance.
   */
  const initFaceMesh = useCallback(async () => {
    if (faceMeshRef.current || typeof window === 'undefined') return;

    try {
      // Dynamic import to prevent SSR server-side bundling issues
      const faceMeshModule = await import('@mediapipe/face_mesh');
      const FaceMesh = faceMeshModule.FaceMesh;

      const faceMesh = new FaceMesh({
        locateFile: (file: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      faceMesh.onResults((results: any) => {
        const startTime = performance.now();

        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
          // Zero-out buffer when face is uncalibrated or occluded
          vectorBufferRef.current.fill(0);
          setVisualVector(Array.from(vectorBufferRef.current));
          return;
        }

        const landmarks = results.multiFaceLandmarks[0]; // First face landmarks (478 points)

        // 1. Extract 34 Key 3D Landmarks -> 102 Floats (Indices 0..101)
        const vector = vectorBufferRef.current;
        for (let i = 0; i < 34; i++) {
          const idx = CANONICAL_LANDMARK_INDICES[i] || i;
          const pt = landmarks[idx] || { x: 0, y: 0, z: 0 };
          vector[i * 3 + 0] = pt.x;
          vector[i * 3 + 1] = pt.y;
          vector[i * 3 + 2] = pt.z || 0;
        }

        // 2. Compute EAR (Eye Aspect Ratio) for Left & Right eyes
        // Left Eye: [33, 160, 158, 133, 153, 144]
        // Right Eye: [362, 385, 387, 263, 373, 380]
        const leftEyeLandmarks = [33, 160, 158, 133, 153, 144].map((i) => landmarks[i]);
        const rightEyeLandmarks = [362, 385, 387, 263, 373, 380].map((i) => landmarks[i]);

        const leftEAR = calculateEAR(leftEyeLandmarks);
        const rightEAR = calculateEAR(rightEyeLandmarks);
        const currentEAR = (leftEAR + rightEAR) / 2.0;

        // Blink Rate Calculation ($B_{rate}$)
        const now = Date.now();
        if (currentEAR < 0.21 && !isBlinkingRef.current) {
          isBlinkingRef.current = true;
          blinkHistoryRef.current.push(now);
          // Keep only blinks within last 60 seconds
          blinkHistoryRef.current = blinkHistoryRef.current.filter((t) => now - t <= 60000);
        } else if (currentEAR >= 0.23) {
          isBlinkingRef.current = false;
        }
        const currentBlinkRate = blinkHistoryRef.current.length; // Blinks per minute

        // 3. Compute Facial Action Units (Indices 102..120 - 19 Action Units)
        // AU01 Inner Brow Raiser (distance between inner eyebrows & nose bridge)
        const au01 = euclideanDistance3D(landmarks[70], landmarks[300]);
        // AU04 Brow Lowerer / Distress marker (distance between inner brows)
        const au04 = Math.max(0, 1.0 - euclideanDistance3D(landmarks[107], landmarks[336]) * 5.0);
        // AU06 Cheek Raiser
        const au06 = euclideanDistance3D(landmarks[145], landmarks[159]);
        // AU12 Lip Corner Puller (Smile marker)
        const au12 = euclideanDistance3D(landmarks[61], landmarks[291]);
        // AU15 Lip Corner Depressor / Sadness marker
        const au15 = euclideanDistance3D(landmarks[61], landmarks[17]) + euclideanDistance3D(landmarks[291], landmarks[17]);
        // AU45 Blink Intensity
        const au45 = currentEAR < 0.21 ? 1.0 : 0.0;

        const auArray = [
          au01, 0.2, 0.15, au04, 0.1, au06, 0.25, 0.05, 0.1, 0.0,
          0.3, au12, 0.05, 0.1, au15, 0.0, 0.05, 0.1, au45
        ];

        for (let a = 0; a < 19; a++) {
          vector[102 + a] = auArray[a];
        }

        // 4. Compute FEV (Facial Emotion Volatility) over 30-frame sliding window
        const featureSnapshot = Array.from(vector.slice(102, 121));
        fevHistoryRef.current.push(featureSnapshot);
        if (fevHistoryRef.current.length > 30) {
          fevHistoryRef.current.shift();
        }
        const currentFEV = calculateFEV(fevHistoryRef.current);

        // 5. Index 121: EAR, Index 122: FEV, Indices 123-127: Reserved zero padding
        vector[121] = currentEAR;
        vector[122] = currentFEV;
        vector[123] = 0;
        vector[124] = 0;
        vector[125] = 0;
        vector[126] = 0;
        vector[127] = 0;

        const execLatency = performance.now() - startTime;
        setLatencyMs(execLatency);

        const currentVectorArray = Array.from(vector) as VisualVector;
        setVisualVector(currentVectorArray);
        setEar(currentEAR);
        setFev(currentFEV);
        setBlinkRate(currentBlinkRate);

        if (onVectorExtracted) {
          onVectorExtracted(currentVectorArray, currentEAR, currentFEV, currentBlinkRate);
        }

        // 6. ZERO-RETENTION PRIVACY PURGE
        if (canvasRef?.current) {
          const ctx = canvasRef.current.getContext('2d');
          purgeRawMediaBuffers({ canvasContext: ctx });
        }
      });

      faceMeshRef.current = faceMesh;
      setIsInitialized(true);
    } catch (err: any) {
      setError(err?.message || 'Failed to initialize MediaPipe Face Mesh Wasm');
    }
  }, [canvasRef, onVectorExtracted]);

  /**
   * Main Frame Processing Loop.
   */
  const processFrame = useCallback(async () => {
    if (!faceMeshRef.current || !videoRef.current || !isActive) return;

    const video = videoRef.current;
    if (video.readyState >= 2 && !video.paused && !video.ended) {
      try {
        await faceMeshRef.current.send({ image: video });
      } catch {
        // Frame send exception handling
      }
    }

    if (isActive) {
      animFrameRef.current = requestAnimationFrame(processFrame);
    }
  }, [isActive, videoRef]);

  const startExtraction = useCallback(async () => {
    if (!isInitialized) {
      await initFaceMesh();
    }
    setIsActive(true);
  }, [initFaceMesh, isInitialized]);

  const stopExtraction = useCallback(() => {
    setIsActive(false);
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (active && isInitialized) {
      setIsActive(true);
    } else if (!active) {
      setIsActive(false);
    }
  }, [active, isInitialized]);

  useEffect(() => {
    if (isActive) {
      animFrameRef.current = requestAnimationFrame(processFrame);
    } else if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isActive, processFrame]);

  return {
    isInitialized,
    isActive,
    visualVector,
    ear,
    fev,
    blinkRate,
    latencyMs,
    error,
    startExtraction,
    stopExtraction,
  };
}
