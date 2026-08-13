import { z } from 'zod';

/**
 * Modality Stream Status: active, degraded, or disabled.
 */
export const ModalityStreamStatusSchema = z.enum(['active', 'degraded', 'disabled']);
export type ModalityStreamStatus = z.infer<typeof ModalityStreamStatusSchema>;

export const ModalityStatusMapSchema = z.object({
  visual: ModalityStreamStatusSchema,
  acoustic: ModalityStreamStatusSchema,
  tabular: ModalityStreamStatusSchema,
});
export type ModalityStatusMap = z.infer<typeof ModalityStatusMapSchema>;

/**
 * Categorical Classification Probabilities Schema (4 Mental Health Classes).
 */
export const ClassificationScoresSchema = z.object({
  healthy: z.number().min(0).max(1),
  mild: z.number().min(0).max(1),
  moderate: z.number().min(0).max(1),
  severe: z.number().min(0).max(1),
  predictedClass: z.enum(['Healthy', 'Mild', 'Moderate', 'Severe']),
});
export type ClassificationScores = z.infer<typeof ClassificationScoresSchema>;

/**
 * Continuous Symptom Regression Scores Schema.
 * Depression (0-34), Anxiety (0-24), Stress (0-39).
 */
export const ContinuousScoresSchema = z.object({
  depression: z.number().min(0).max(34),
  anxiety: z.number().min(0).max(24),
  stress: z.number().min(0).max(39),
  confidenceMargin: z.number().min(0).default(2.5),
});
export type ContinuousScores = z.infer<typeof ContinuousScoresSchema>;

/**
 * FastSHAP Feature Attribution Entry.
 */
export const FastShapAttributionSchema = z.object({
  featureName: z.string(),
  shapValue: z.number(), // +phi_i (risk contributor) or -phi_i (resilience contributor)
  category: z.enum(['behavioral', 'visual', 'acoustic', 'physiological']),
});
export type FastShapAttribution = z.infer<typeof FastShapAttributionSchema>;

/**
 * Action Unit Spatial Saliency Heatmap Weights.
 */
export const SaliencyWeightsSchema = z.object({
  au04BrowLowerer: z.number().min(0).max(1), // Distress marker
  au15LipDepressor: z.number().min(0).max(1), // Sad affect marker
  au06CheekRaiser: z.number().min(0).max(1),
  au12SmilePuller: z.number().min(0).max(1),
  contourMesh: z.array(z.number()).optional(),
});
export type SaliencyWeights = z.infer<typeof SaliencyWeightsSchema>;

/**
 * Complete Backend Inference Evaluation Response Frame.
 */
export const BackendInferenceResponseSchema = z.object({
  sequenceId: z.number().int().nonnegative(),
  timestamp: z.number(),
  transitLatencyMs: z.number().nonnegative().default(18),
  classification: ClassificationScoresSchema,
  continuousScores: ContinuousScoresSchema,
  fastShapAttributions: z.array(FastShapAttributionSchema).length(18),
  saliencyWeights: SaliencyWeightsSchema,
  clinicalNarrative: z.string(),
  modalityStatus: ModalityStatusMapSchema,
});
export type BackendInferenceResponse = z.infer<typeof BackendInferenceResponseSchema>;

/**
 * WebSocket Connection States.
 */
export type ConnectionState = 'DISCONNECTED' | 'CONNECTING' | 'CONNECTED' | 'RECONNECTING';
