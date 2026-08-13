import { useCallback, useEffect, useRef, useState } from 'react';
import { AcousticVector, PAYLOAD_CONSTANTS } from '../types/payload';
import { purgeRawMediaBuffers } from '../utils/privacyManager';

export interface UseAudioProcessorOptions {
  active?: boolean;
  onVectorExtracted?: (vector: AcousticVector) => void;
}

export interface UseAudioProcessorReturn {
  isWorkletLoaded: boolean;
  isAudioActive: boolean;
  acousticVector: AcousticVector | null;
  volumeLevel: number;
  error: string | null;
  startAudio: () => Promise<void>;
  stopAudio: () => void;
}

export function useAudioProcessor({
  active = false,
  onVectorExtracted,
}: UseAudioProcessorOptions = {}): UseAudioProcessorReturn {
  const [isWorkletLoaded, setIsWorkletLoaded] = useState(false);
  const [isAudioActive, setIsAudioActive] = useState(false);
  const [acousticVector, setAcousticVector] = useState<AcousticVector | null>(null);
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);

  /**
   * Initializes Web Audio API Context and registers the AudioWorklet processor module.
   */
  const initAudioWorklet = useCallback(async () => {
    if (isWorkletLoaded && audioContextRef.current) return;

    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: 16000 });

      // Ensure AudioContext is resumed if suspended by browser autoplay policies
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      // Register off-main-thread Worklet module from public directory
      await audioCtx.audioWorklet.addModule('/audio-processor.worklet.js');

      audioContextRef.current = audioCtx;
      setIsWorkletLoaded(true);
    } catch (err: any) {
      setError(err?.message || 'Failed to register AudioWorklet module');
    }
  }, [isWorkletLoaded]);

  /**
   * Requests microphone permissions and connects audio graph to worklet node.
   */
  const startAudio = useCallback(async () => {
    try {
      setError(null);
      if (!audioContextRef.current) {
        await initAudioWorklet();
      }

      const audioCtx = audioContextRef.current;
      if (!audioCtx) throw new Error('AudioContext initialization failed');

      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      // 1. Request microphone input stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
        },
        video: false,
      });

      streamRef.current = stream;

      // 2. Create MediaStreamSourceNode
      const sourceNode = audioCtx.createMediaStreamSource(stream);
      sourceNodeRef.current = sourceNode;

      // 3. Create AudioWorkletNode
      const workletNode = new AudioWorkletNode(audioCtx, 'audio-processor-worklet');
      workletNodeRef.current = workletNode;

      // 4. Handle incoming zero-copy transferred Float32Array vectors from worklet thread
      workletNode.port.onmessage = (event: MessageEvent) => {
        if (event.data?.type === 'ACOUSTIC_VECTOR' && event.data.vector) {
          const rawBuffer = event.data.vector as ArrayBuffer;
          const floatArray = new Float32Array(rawBuffer);

          if (floatArray.length === PAYLOAD_CONSTANTS.ACOUSTIC_VECTOR_LENGTH) {
            const vectorArray = Array.from(floatArray) as AcousticVector;
            setAcousticVector(vectorArray);

            // Compute volume metric for UI indicators (RMS Energy)
            const rms = floatArray[0] || 0;
            setVolumeLevel(Math.min(1.0, rms * 5.0));

            if (onVectorExtracted) {
              onVectorExtracted(vectorArray);
            }

            // Zero-retention privacy cleanup on raw buffer
            purgeRawMediaBuffers({ audioBuffer: floatArray });
          }
        }
      };

      // Connect audio graph: Mic Source -> Worklet Node -> (silent destination or disconnect)
      sourceNode.connect(workletNode);
      setIsAudioActive(true);
    } catch (err: any) {
      setError(err?.message || 'Microphone access denied or audio graph connection failed');
      setIsAudioActive(false);
    }
  }, [initAudioWorklet, onVectorExtracted]);

  /**
   * Disconnects audio nodes and stops microphone media stream tracks.
   */
  const stopAudio = useCallback(() => {
    if (sourceNodeRef.current && workletNodeRef.current) {
      try {
        sourceNodeRef.current.disconnect(workletNodeRef.current);
      } catch {
        // Disconnect cleanup error handling
      }
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.suspend();
    }

    setIsAudioActive(false);
    setVolumeLevel(0);
  }, []);

  useEffect(() => {
    if (active && !isAudioActive) {
      startAudio();
    } else if (!active && isAudioActive) {
      stopAudio();
    }
  }, [active, isAudioActive, startAudio, stopAudio]);

  useEffect(() => {
    return () => {
      stopAudio();
    };
  }, [stopAudio]);

  return {
    isWorkletLoaded,
    isAudioActive,
    acousticVector,
    volumeLevel,
    error,
    startAudio,
    stopAudio,
  };
}
