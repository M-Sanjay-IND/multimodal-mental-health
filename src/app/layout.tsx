import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Psychiatric Evaluation Dashboard',
  description: 'Multimodal Psychiatric Evaluation & Affective Assessment Dashboard',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="light antialiased">
      <head>
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap"
        />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&family=Geist:wght@400;600;700&display=swap"
        />
      </head>
      <body className="bg-background text-on-background min-h-screen overflow-y-auto antialiased">
        {children}
      </body>
    </html>
  );
}

