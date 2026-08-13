import { z } from 'zod';

/**
 * Constants defining the exact payload dimensions and binary buffer offsets.
 */
export const PAYLOAD_CONSTANTS = {
  VISUAL_VECTOR_LENGTH: 128,
  ACOUSTIC_VECTOR_LENGTH: 256,
  TABULAR_VECTOR_LENGTH: 18,

  HEADER_BYTE_SIZE: 24,
  VISUAL_BYTE_SIZE: 512, // 128 * 4
  ACOUSTIC_BYTE_SIZE: 1024, // 256 * 4
  TABULAR_BYTE_SIZE: 72, // 18 * 4

  TOTAL_BUFFER_BYTE_SIZE: 1632, // 24 + 512 + 1024 + 72 (offsets 0..1631)
} as const;

/**
 * Zod refinement helper for finite float arrays.
 * Rejects arrays with non-numeric values, NaN, Infinity, or -Infinity.
 */
const finiteFloatArray = (expectedLength: number, name: string) =>
  z
    .array(z.number().finite({ message: `${name} elements must be finite numbers (no NaN or Infinity)` }))
    .length(expectedLength, { message: `${name} must contain exactly ${expectedLength} elements` });

/**
 * Modality Status Flags schema (4 Uint8 flags: visual, acoustic, tabular, reserved).
 */
export const ModalityStatusFlagsSchema = z.object({
  visualActive: z.number().int().min(0).max(1),
  acousticActive: z.number().int().min(0).max(1),
  tabularActive: z.number().int().min(0).max(1),
  reserved: z.number().int().min(0).max(255).default(0),
});

export type ModalityStatusFlags = z.infer<typeof ModalityStatusFlagsSchema>;

/**
 * Binary Header metadata schema.
 */
export const BinaryHeaderSchema = z.object({
  sequenceId: z.number().int().nonnegative({ message: 'Sequence ID must be a non-negative integer' }),
  sessionIdHash: z.number().int().nonnegative({ message: 'Session ID Hash must be a non-negative integer' }),
  timestamp: z.number().finite({ message: 'Timestamp must be a valid finite number' }),
  flags: ModalityStatusFlagsSchema,
});

export type BinaryHeader = z.infer<typeof BinaryHeaderSchema>;

/**
 * Visual Vector ($E_V^{edge}$) Schema - 128 dimensions.
 * Indices 0-101: 34 3D landmark coordinates (x, y, z)
 * Indices 102-120: 19 Facial Action Unit continuous intensities
 * Index 121: Eye Aspect Ratio (EAR)
 * Index 122: Facial Emotion Volatility (FEV)
 * Indices 123-127: Reserved zero padding
 */
export const VisualVectorSchema = finiteFloatArray(
  PAYLOAD_CONSTANTS.VISUAL_VECTOR_LENGTH,
  'Visual Vector'
);

export type VisualVector = z.infer<typeof VisualVectorSchema>;

/**
 * Acoustic Vector ($E_A^{edge}$) Schema - 256 dimensions.
 * 88 eGeMAPS continuous descriptors + fine-tuned wav2vec2 latent features.
 */
export const AcousticVectorSchema = finiteFloatArray(
  PAYLOAD_CONSTANTS.ACOUSTIC_VECTOR_LENGTH,
  'Acoustic Vector'
);

export type AcousticVector = z.infer<typeof AcousticVectorSchema>;

/**
 * Tabular Vector ($\mathbf{x}_{tab}$) Schema - 18 continuous metrics.
 * 1. Sleep_Quality (1-5)
 * 2. Social_Engagement (1-5)
 * 3. Daily_App_Usage_Min
 * 4. Typing_Speed_WPM
 * 5. Session_Frequency
 * 6. Idle_Time_Min
 * 7. Facial_Emotion_Variance
 * 8. Eye_Blink_Rate
 * 9. Smile_Intensity
 * 10. Head_Motion_Index
 * 11. MFCC_Mean
 * 12. MFCC_Variance
 * 13. Pitch_Mean
 * 14. Speech_Rate
 * 15. Heart_Rate_BPM
 * 16. HRV_Index
 * 17. Skin_Temperature
 * 18. GSR_Level
 */
export const TabularVectorSchema = finiteFloatArray(
  PAYLOAD_CONSTANTS.TABULAR_VECTOR_LENGTH,
  'Tabular Vector'
);

export type TabularVector = z.infer<typeof TabularVectorSchema>;

/**
 * Complete Unified Client Payload Zod Schema.
 * Used to validate raw uncompressed data before serialization.
 */
export const UnifiedClientPayloadSchema = z.object({
  header: BinaryHeaderSchema,
  visualVector: VisualVectorSchema,
  acousticVector: AcousticVectorSchema,
  tabularVector: TabularVectorSchema,
});

export type UnifiedClientPayload = z.infer<typeof UnifiedClientPayloadSchema>;
