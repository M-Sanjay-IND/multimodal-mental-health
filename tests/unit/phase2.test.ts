import { describe, expect, it, vi } from 'vitest';
import {
  calculateEAR,
  calculateFEV,
  euclideanDistance3D,
} from '../../src/hooks/useMediaPipeFaceMesh';
import { overwriteBufferWithZeros, purgeRawMediaBuffers } from '../../src/utils/privacyManager';

describe('Phase 2: Edge Feature Extraction & Zero-Retention Privacy Suite', () => {
  describe('3D Geometric Metrics & Feature Extraction Math', () => {
    it('should correctly compute 3D Euclidean distances', () => {
      const p1 = { x: 0, y: 0, z: 0 };
      const p2 = { x: 3, y: 4, z: 0 };
      expect(euclideanDistance3D(p1, p2)).toBeCloseTo(5.0, 5);

      const p3 = { x: 1, y: 2, z: 2 };
      const p4 = { x: 4, y: 6, z: 2 };
      expect(euclideanDistance3D(p3, p4)).toBeCloseTo(5.0, 5);
    });

    it('should compute valid Eye Aspect Ratio (EAR)', () => {
      const eyeOpenLandmarks = [
        { x: 0.1, y: 0.5 },
        { x: 0.2, y: 0.4 },
        { x: 0.3, y: 0.4 },
        { x: 0.4, y: 0.5 },
        { x: 0.3, y: 0.6 },
        { x: 0.2, y: 0.6 },
      ];

      const earOpen = calculateEAR(eyeOpenLandmarks);
      expect(earOpen).toBeGreaterThan(0.2);

      const eyeClosedLandmarks = [
        { x: 0.1, y: 0.5 },
        { x: 0.2, y: 0.5 },
        { x: 0.3, y: 0.5 },
        { x: 0.4, y: 0.5 },
        { x: 0.3, y: 0.5 },
        { x: 0.2, y: 0.5 },
      ];

      const earClosed = calculateEAR(eyeClosedLandmarks);
      expect(earClosed).toBeCloseTo(0.0, 5);
    });

    it('should compute Facial Emotion Volatility (FEV) over sliding window W=30', () => {
      // 30 identical static frames should yield 0 volatility
      const staticHistory = new Array(30).fill([0.5, 0.2, 0.8, 0.1]);
      expect(calculateFEV(staticHistory)).toBeCloseTo(0.0, 5);

      // High variance history should yield non-zero FEV
      const dynamicHistory: number[][] = [];
      for (let i = 0; i < 30; i++) {
        dynamicHistory.push([Math.sin(i), Math.cos(i), i * 0.1, (i % 2) * 0.5]);
      }
      const dynamicFEV = calculateFEV(dynamicHistory);
      expect(dynamicFEV).toBeGreaterThan(0.1);
    });
  });

  describe('Zero-Retention Privacy Manager & Buffer Sanitization', () => {
    it('should overwrite TypedArrays in-place with zeros', () => {
      const pcmBuffer = new Float32Array([0.5, -0.2, 0.8, 1.2, -0.9]);
      overwriteBufferWithZeros(pcmBuffer);

      for (let i = 0; i < pcmBuffer.length; i++) {
        expect(pcmBuffer[i]).toBe(0);
      }
    });

    it('should overwrite ArrayBuffers in-place with zeros', () => {
      const buffer = new Float32Array([1.0, 2.0, 3.0]).buffer;
      overwriteBufferWithZeros(buffer);

      const view = new Float32Array(buffer);
      expect(view[0]).toBe(0);
      expect(view[1]).toBe(0);
      expect(view[2]).toBe(0);
    });

    it('should purge Canvas pixels and audio PCM buffers without throwing exceptions', () => {
      const pcmBuffer = new Float32Array([0.1, 0.2, 0.3]);

      // Mock canvas context
      const mockContext = {
        canvas: { width: 100, height: 100 },
        clearRect: vi.fn(),
        getImageData: vi.fn().mockReturnValue({
          data: new Uint8ClampedArray(40000).fill(255),
        }),
        putImageData: vi.fn(),
      } as unknown as CanvasRenderingContext2D;

      expect(() =>
        purgeRawMediaBuffers({
          canvasContext: mockContext,
          audioBuffer: pcmBuffer,
        })
      ).not.toThrow();

      expect(mockContext.clearRect).toHaveBeenCalledWith(0, 0, 100, 100);
      expect(mockContext.getImageData).toHaveBeenCalledWith(0, 0, 100, 100);

      // Verify audio buffer zeroed
      expect(pcmBuffer[0]).toBe(0);
      expect(pcmBuffer[1]).toBe(0);
      expect(pcmBuffer[2]).toBe(0);
    });
  });
});
