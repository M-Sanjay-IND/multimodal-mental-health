import {
  PAYLOAD_CONSTANTS,
  UnifiedClientPayload,
  UnifiedClientPayloadSchema,
} from '../types/payload';

/**
 * Computes a deterministic 32-bit unsigned integer hash from a Session ID string (e.g. UUIDv4).
 */
export function hashSessionId(sessionId: string): number {
  let hash = 5381;
  for (let i = 0; i < sessionId.length; i++) {
    hash = (hash * 33) ^ sessionId.charCodeAt(i);
  }
  return hash >>> 0; // Convert to 32-bit unsigned int
}

/**
 * Serializes a UnifiedClientPayload object into a 1,624-byte contiguous ArrayBuffer.
 *
 * Buffer Offset Layout (1,624 Bytes Total):
 * - Bytes 0..3:    Sequence ID (Uint32)
 * - Bytes 4..7:    Session ID Hash (Uint32)
 * - Bytes 8..15:   Timestamp (Float64)
 * - Bytes 16..19:  Flags (4 x Uint8: visualActive, acousticActive, tabularActive, reserved)
 * - Bytes 20..23:  Header Padding (4 x Uint8 zeros)
 * - Bytes 24..535: Visual Vector E_V (128 x Float32 = 512 Bytes)
 * - Bytes 536..1559: Acoustic Vector E_A (256 x Float32 = 1024 Bytes)
 * - Bytes 1560..1631: Tabular Vector x_tab (18 x Float32 = 72 Bytes)
 */
export function serializeToBinaryBuffer(payload: UnifiedClientPayload): ArrayBuffer {
  // 1. Validate payload against Zod schema (ensures valid lengths, no NaN/Infinity)
  const validatedPayload = UnifiedClientPayloadSchema.parse(payload);

  const buffer = new ArrayBuffer(PAYLOAD_CONSTANTS.TOTAL_BUFFER_BYTE_SIZE);
  const dataView = new DataView(buffer);

  // 2. Write Binary Header (Bytes 0 to 23)
  dataView.setUint32(0, validatedPayload.header.sequenceId, true); // Little-endian
  dataView.setUint32(4, validatedPayload.header.sessionIdHash, true);
  dataView.setFloat64(8, validatedPayload.header.timestamp, true);

  // Flags (Bytes 16..19)
  dataView.setUint8(16, validatedPayload.header.flags.visualActive);
  dataView.setUint8(17, validatedPayload.header.flags.acousticActive);
  dataView.setUint8(18, validatedPayload.header.flags.tabularActive);
  dataView.setUint8(19, validatedPayload.header.flags.reserved);

  // Header Padding (Bytes 20..23)
  dataView.setUint32(20, 0, true);

  // 3. Write Visual Vector E_V (Bytes 24 to 535)
  const visualOffset = 24;
  for (let i = 0; i < PAYLOAD_CONSTANTS.VISUAL_VECTOR_LENGTH; i++) {
    dataView.setFloat32(visualOffset + i * 4, validatedPayload.visualVector[i], true);
  }

  // 4. Write Acoustic Vector E_A (Bytes 536 to 1559)
  const acousticOffset = 536;
  for (let i = 0; i < PAYLOAD_CONSTANTS.ACOUSTIC_VECTOR_LENGTH; i++) {
    dataView.setFloat32(acousticOffset + i * 4, validatedPayload.acousticVector[i], true);
  }

  // 5. Write Tabular Vector x_tab (Bytes 1560 to 1631)
  const tabularOffset = 1560;
  for (let i = 0; i < PAYLOAD_CONSTANTS.TABULAR_VECTOR_LENGTH; i++) {
    dataView.setFloat32(tabularOffset + i * 4, validatedPayload.tabularVector[i], true);
  }

  return buffer;
}

/**
 * Deserializes a 1,624-byte contiguous ArrayBuffer back into a validated UnifiedClientPayload.
 */
export function deserializeFromBinaryBuffer(buffer: ArrayBuffer): UnifiedClientPayload {
  if (buffer.byteLength !== PAYLOAD_CONSTANTS.TOTAL_BUFFER_BYTE_SIZE) {
    throw new Error(
      `Invalid buffer byte length: expected ${PAYLOAD_CONSTANTS.TOTAL_BUFFER_BYTE_SIZE} bytes, received ${buffer.byteLength} bytes`
    );
  }

  const dataView = new DataView(buffer);

  // 1. Parse Binary Header
  const sequenceId = dataView.getUint32(0, true);
  const sessionIdHash = dataView.getUint32(4, true);
  const timestamp = dataView.getFloat64(8, true);

  const visualActive = dataView.getUint8(16);
  const acousticActive = dataView.getUint8(17);
  const tabularActive = dataView.getUint8(18);
  const reserved = dataView.getUint8(19);

  // 2. Parse Visual Vector E_V (128 Float32s)
  const visualVector: number[] = new Array(PAYLOAD_CONSTANTS.VISUAL_VECTOR_LENGTH);
  const visualOffset = 24;
  for (let i = 0; i < PAYLOAD_CONSTANTS.VISUAL_VECTOR_LENGTH; i++) {
    visualVector[i] = dataView.getFloat32(visualOffset + i * 4, true);
  }

  // 3. Parse Acoustic Vector E_A (256 Float32s)
  const acousticVector: number[] = new Array(PAYLOAD_CONSTANTS.ACOUSTIC_VECTOR_LENGTH);
  const acousticOffset = 536;
  for (let i = 0; i < PAYLOAD_CONSTANTS.ACOUSTIC_VECTOR_LENGTH; i++) {
    acousticVector[i] = dataView.getFloat32(acousticOffset + i * 4, true);
  }

  // 4. Parse Tabular Vector x_tab (18 Float32s)
  const tabularVector: number[] = new Array(PAYLOAD_CONSTANTS.TABULAR_VECTOR_LENGTH);
  const tabularOffset = 1560;
  for (let i = 0; i < PAYLOAD_CONSTANTS.TABULAR_VECTOR_LENGTH; i++) {
    tabularVector[i] = dataView.getFloat32(tabularOffset + i * 4, true);
  }

  const payload: UnifiedClientPayload = {
    header: {
      sequenceId,
      sessionIdHash,
      timestamp,
      flags: {
        visualActive,
        acousticActive,
        tabularActive,
        reserved,
      },
    },
    visualVector,
    acousticVector,
    tabularVector,
  };

  // 5. Return validated payload
  return UnifiedClientPayloadSchema.parse(payload);
}
