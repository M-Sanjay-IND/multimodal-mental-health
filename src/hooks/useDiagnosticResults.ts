import { useContext, useMemo } from 'react';
import {
  DiagnosticContext,
  DiagnosticContextState,
  ScoreHistoryPoint,
} from '../context/DiagnosticContext';
import { FastShapAttribution } from '../types/response';

export type { ScoreHistoryPoint };

export interface ExtendedDiagnosticResults extends DiagnosticContextState {
  topRiskFactors: FastShapAttribution[];
  topResilienceFactors: FastShapAttribution[];
  isHealthy: boolean;
  isHighRisk: boolean;
}

/**
 * Custom React Hook consuming diagnostic evaluation results, FastSHAP rankings,
 * connection status, and networking actions.
 */
export function useDiagnosticResults(): ExtendedDiagnosticResults {
  const context = useContext(DiagnosticContext);

  if (!context) {
    throw new Error('useDiagnosticResults must be used within a DiagnosticProvider');
  }

  // Derived FastSHAP attributions: Top 3 Risk (+phi_i) & Top 2 Resilience (-phi_i)
  const topRiskFactors = useMemo(() => {
    return [...context.fastShapAttributions]
      .filter((attr) => attr.shapValue > 0)
      .sort((a, b) => b.shapValue - a.shapValue)
      .slice(0, 3);
  }, [context.fastShapAttributions]);

  const topResilienceFactors = useMemo(() => {
    return [...context.fastShapAttributions]
      .filter((attr) => attr.shapValue < 0)
      .sort((a, b) => a.shapValue - b.shapValue)
      .slice(0, 2);
  }, [context.fastShapAttributions]);

  const isHealthy = context.classification.predictedClass === 'Healthy';
  const isHighRisk =
    context.classification.predictedClass === 'Severe' ||
    context.classification.predictedClass === 'Moderate';

  return {
    ...context,
    topRiskFactors,
    topResilienceFactors,
    isHealthy,
    isHighRisk,
  };
}
