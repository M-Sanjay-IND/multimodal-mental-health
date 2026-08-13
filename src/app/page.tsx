'use client';

import { DiagnosticProvider } from '../context/DiagnosticContext';
import { Dashboard } from '../components/Dashboard';

export default function Home() {
  return (
    <DiagnosticProvider autoConnect={false}>
      <Dashboard />
    </DiagnosticProvider>
  );
}
