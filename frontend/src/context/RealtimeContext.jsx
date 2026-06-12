import React, { createContext, useContext } from 'react';
import useRealtime from '../hooks/useRealtime';

const RealtimeContext = createContext(null);

export function RealtimeProvider({ children }) {
  const realtime = useRealtime();
  return (
    <RealtimeContext.Provider value={realtime}>
      {children}
    </RealtimeContext.Provider>
  );
}

export function useRealtimeContext() {
  const ctx = useContext(RealtimeContext);
  if (!ctx) throw new Error('useRealtimeContext must be used inside RealtimeProvider');
  return ctx;
}
