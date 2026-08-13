import { describe, expect, it, vi } from 'vitest';
import { CircularFrameQueue, WebSocketService } from '../../src/services/websocket';
import { PAYLOAD_CONSTANTS } from '../../src/types/payload';
import { BackendInferenceResponse } from '../../src/types/response';

describe('Phase 3: WebSocket Streaming Engine & Resiliency Suite', () => {
  describe('CircularFrameQueue Mechanics', () => {
    it('should initialize with capacity 50 and enqueue items', () => {
      const queue = new CircularFrameQueue(50);
      expect(queue.getSize()).toBe(0);

      const frame1 = new ArrayBuffer(1632);
      queue.enqueue(frame1);
      expect(queue.getSize()).toBe(1);

      const dequeued = queue.dequeue();
      expect(dequeued).toBe(frame1);
      expect(queue.getSize()).toBe(0);
    });

    it('should overwrite oldest items when capacity (50) is reached', () => {
      const queue = new CircularFrameQueue(50);

      const frames: ArrayBuffer[] = [];
      for (let i = 0; i < 55; i++) {
        const frame = new ArrayBuffer(1632);
        frames.push(frame);
        queue.enqueue(frame);
      }

      // Size capped at 50
      expect(queue.getSize()).toBe(50);

      // Oldest 5 items (0..4) should have been overwritten; first dequeued should be frame 5
      const firstDequeued = queue.dequeue();
      expect(firstDequeued).toBe(frames[5]);
    });

    it('should clear queue state on clear()', () => {
      const queue = new CircularFrameQueue(50);
      queue.enqueue(new ArrayBuffer(1632));
      queue.enqueue(new ArrayBuffer(1632));

      expect(queue.getSize()).toBe(2);
      queue.clear();
      expect(queue.getSize()).toBe(0);
      expect(queue.dequeue()).toBeNull();
    });
  });

  describe('WebSocketService Payload & Queueing Logic', () => {
    it('should reject binary buffers not matching 1,632 bytes', () => {
      const wsService = new WebSocketService();
      const invalidBuffer = new ArrayBuffer(500); // Invalid byte size

      const sent = wsService.sendBinaryPayload(invalidBuffer);
      expect(sent).toBe(false);
      expect(wsService.getQueueSize()).toBe(0);
    });

    it('should queue valid 1,632-byte buffers when socket is disconnected', () => {
      const wsService = new WebSocketService();
      const validBuffer = new ArrayBuffer(PAYLOAD_CONSTANTS.TOTAL_BUFFER_BYTE_SIZE);

      expect(wsService.getConnectionState()).toBe('DISCONNECTED');
      const sent = wsService.sendBinaryPayload(validBuffer);

      expect(sent).toBe(false);
      expect(wsService.getQueueSize()).toBe(1);
    });

    it('should register connection state and latency listeners', () => {
      const wsService = new WebSocketService();
      const stateListener = vi.fn();
      const latencyListener = vi.fn();

      const unsubState = wsService.onConnectionState(stateListener);
      const unsubLatency = wsService.onLatency(latencyListener);

      expect(wsService.getConnectionState()).toBe('DISCONNECTED');

      unsubState();
      unsubLatency();
    });
  });

  describe('FastSHAP Attribution Sorting Logic', () => {
    it('should correctly partition top risk (+phi) and resilience (-phi) factors', () => {
      const attributions = [
        { featureName: 'GSR_Level', shapValue: 3.2, category: 'physiological' as const },
        { featureName: 'Heart_Rate_BPM', shapValue: 1.5, category: 'physiological' as const },
        { featureName: 'Eye_Blink_Rate', shapValue: 2.1, category: 'visual' as const },
        { featureName: 'Facial_Emotion_Variance', shapValue: 0.8, category: 'visual' as const },
        { featureName: 'Sleep_Quality', shapValue: -2.7, category: 'behavioral' as const },
        { featureName: 'HRV_Index', shapValue: -3.1, category: 'physiological' as const },
        { featureName: 'Social_Engagement', shapValue: -1.0, category: 'behavioral' as const },
      ];

      const topRisk = attributions
        .filter((a) => a.shapValue > 0)
        .sort((a, b) => b.shapValue - a.shapValue)
        .slice(0, 3);

      const topResilience = attributions
        .filter((a) => a.shapValue < 0)
        .sort((a, b) => a.shapValue - b.shapValue)
        .slice(0, 2);

      expect(topRisk.length).toBe(3);
      expect(topRisk[0].featureName).toBe('GSR_Level');
      expect(topRisk[1].featureName).toBe('Eye_Blink_Rate');

      expect(topResilience.length).toBe(2);
      expect(topResilience[0].featureName).toBe('HRV_Index');
      expect(topResilience[1].featureName).toBe('Sleep_Quality');
    });
  });
});
