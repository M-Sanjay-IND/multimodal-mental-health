import { describe, expect, it } from 'vitest';
import {
  PAYLOAD_CONSTANTS,
  UnifiedClientPayload,
  UnifiedClientPayloadSchema,
} from '../../src/types/payload';
import {
  deserializeFromBinaryBuffer,
  hashSessionId,
  serializeToBinaryBuffer,
} from '../../src/utils/serialization';

describe('Phase 1: Binary Serialization & Data Contract Suite', () => {
  const createDummyPayload = (
    overrides?: Partial<UnifiedClientPayload>
  ): UnifiedClientPayload => {
    const visualVector = new Array(128).fill(0).map((_, i) => i * 0.01);
    const acousticVector = new Array(256).fill(0).map((_, i) => -1.0 + i * 0.005);
    const tabularVector = [
      4.0, // Sleep_Quality
      3.0, // Social_Engagement
      120.0, // Daily_App_Usage_Min
      65.0, // Typing_Speed_WPM
      15.0, // Session_Frequency
      45.0, // Idle_Time_Min
      0.15, // Facial_Emotion_Variance
      18.0, // Eye_Blink_Rate
      0.75, // Smile_Intensity
      0.42, // Head_Motion_Index
      -12.4, // MFCC_Mean
      4.2, // MFCC_Variance
      142.5, // Pitch_Mean
      2.1, // Speech_Rate
      72.0, // Heart_Rate_BPM
      45.0, // HRV_Index
      34.2, // Skin_Temperature
      8.5, // GSR_Level
    ];

    return {
      header: {
        sequenceId: 1042,
        sessionIdHash: hashSessionId('session-uuid-v4-abc-123'),
        timestamp: 1776075600000.0,
        flags: {
          visualActive: 1,
          acousticActive: 1,
          tabularActive: 1,
          reserved: 0,
        },
      },
      visualVector,
      acousticVector,
      tabularVector,
      ...overrides,
    };
  };

  it('should verify exact byte length of 1,632 bytes upon serialization', () => {
    const payload = createDummyPayload();
    const buffer = serializeToBinaryBuffer(payload);

    expect(buffer.byteLength).toBe(1632);
    expect(PAYLOAD_CONSTANTS.TOTAL_BUFFER_BYTE_SIZE).toBe(1632);
  });

  it('should achieve 100% round-trip serialization and deserialization fidelity', () => {
    const originalPayload = createDummyPayload();
    const buffer = serializeToBinaryBuffer(originalPayload);
    const deserialized = deserializeFromBinaryBuffer(buffer);

    expect(deserialized.header.sequenceId).toBe(originalPayload.header.sequenceId);
    expect(deserialized.header.sessionIdHash).toBe(originalPayload.header.sessionIdHash);
    expect(deserialized.header.timestamp).toBe(originalPayload.header.timestamp);
    expect(deserialized.header.flags).toEqual(originalPayload.header.flags);

    // Verify Float32 arrays precision match
    for (let i = 0; i < 128; i++) {
      expect(deserialized.visualVector[i]).toBeCloseTo(originalPayload.visualVector[i], 5);
    }
    for (let i = 0; i < 256; i++) {
      expect(deserialized.acousticVector[i]).toBeCloseTo(originalPayload.acousticVector[i], 5);
    }
    for (let i = 0; i < 18; i++) {
      expect(deserialized.tabularVector[i]).toBeCloseTo(originalPayload.tabularVector[i], 5);
    }
  });

  it('should verify precise header and feature segment byte offsets in ArrayBuffer', () => {
    const payload = createDummyPayload({
      header: {
        sequenceId: 0x12345678,
        sessionIdHash: 0x9abcdef0,
        timestamp: 1000.0,
        flags: { visualActive: 1, acousticActive: 0, tabularActive: 1, reserved: 0 },
      },
    });

    const buffer = serializeToBinaryBuffer(payload);
    const view = new DataView(buffer);

    // Byte 0-3: Sequence ID
    expect(view.getUint32(0, true)).toBe(0x12345678);
    // Byte 4-7: Session ID Hash
    expect(view.getUint32(4, true)).toBe(0x9abcdef0);
    // Byte 8-15: Timestamp
    expect(view.getFloat64(8, true)).toBe(1000.0);
    // Byte 16-19: Flags
    expect(view.getUint8(16)).toBe(1);
    expect(view.getUint8(17)).toBe(0);
    expect(view.getUint8(18)).toBe(1);
    expect(view.getUint8(19)).toBe(0);

    // Byte 24: Visual Vector start offset
    expect(view.getFloat32(24, true)).toBeCloseTo(payload.visualVector[0], 5);
    // Byte 536: Acoustic Vector start offset
    expect(view.getFloat32(536, true)).toBeCloseTo(payload.acousticVector[0], 5);
    // Byte 1560: Tabular Vector start offset
    expect(view.getFloat32(1560, true)).toBeCloseTo(payload.tabularVector[0], 5);
  });

  it('should reject non-finite numbers (NaN, Infinity, -Infinity) during Zod validation', () => {
    const invalidPayloadWithNaN = createDummyPayload();
    invalidPayloadWithNaN.visualVector[5] = NaN;

    expect(() => serializeToBinaryBuffer(invalidPayloadWithNaN)).toThrow();

    const invalidPayloadWithInfinity = createDummyPayload();
    invalidPayloadWithInfinity.tabularVector[2] = Infinity;

    expect(() => serializeToBinaryBuffer(invalidPayloadWithInfinity)).toThrow();
  });

  it('should reject invalid array dimensions (e.g. 127 visual features)', () => {
    const payloadShortVisual = createDummyPayload();
    payloadShortVisual.visualVector.pop(); // length 127

    expect(() => serializeToBinaryBuffer(payloadShortVisual)).toThrow(
      /Visual Vector must contain exactly 128 elements/
    );
  });

  it('should reject buffers with invalid byte length during deserialization', () => {
    const invalidBuffer = new ArrayBuffer(1000); // Expected 1,624 bytes
    expect(() => deserializeFromBinaryBuffer(invalidBuffer)).toThrow(
      /Invalid buffer byte length: expected 1632 bytes, received 1000 bytes/
    );
  });

  it('should compute deterministic 32-bit unsigned hashes for session IDs', () => {
    const hash1 = hashSessionId('session-uuid-v4-abc-123');
    const hash2 = hashSessionId('session-uuid-v4-abc-123');
    const hash3 = hashSessionId('different-session-uuid');

    expect(hash1).toBe(hash2);
    expect(hash1).not.toBe(hash3);
    expect(hash1).toBeGreaterThanOrEqual(0);
  });
});
