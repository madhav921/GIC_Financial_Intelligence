import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { can } from './permissions';

function Splash() {
  return (
    <div
      className="flex flex-col items-center justify-center h-screen w-full gap-4"
      style={{ backgroundColor: '#0f172a' }}
    >
      <div className="text-3xl">📡</div>
      <div className="text-slate-300 text-sm font-medium tracking-wide">
        Initializing GIC Intelligence…
      </div>
      <div className="w-40 h-1 rounded-full overflow-hidden bg-slate-700">
        <div
          className="h-full bg-blue-500"
          style={{ width: '40%', animation: 'gicSplash 1.2s ease-in-out infinite' }}
        />
      </div>
      <style>{`@keyframes gicSplash {0%{transform:translateX(-100%)}100%{transform:translateX(350%)}}`}</style>
    </div>
  );
}

function AccessRestricted({ user, permission }) {
  return (
    <div className="flex items-center justify-center w-full" style={{ minHeight: '60vh' }}>
      <div className="max-w-md w-full rounded-xl border border-slate-700 bg-slate-800 p-8 text-center shadow-xl">
        <div className="text-4xl mb-4">🔒</div>
        <h2 className="text-xl font-bold text-white mb-2">Access Restricted</h2>
        <p className="text-slate-400 text-sm mb-4">
          Your current role
          {user?.role ? (
            <span className="mx-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-slate-700 text-slate-200">
              {user.role}
            </span>
          ) : (
            ' '
          )}
          does not have permission to view this section.
        </p>
        <div className="text-xs text-slate-500 font-mono bg-slate-900/60 rounded-lg px-3 py-2 border border-slate-700">
          required: {permission}
        </div>
        <p className="text-xs text-slate-500 mt-4">
          Contact an administrator to request elevated access.
        </p>
      </div>
    </div>
  );
}

export default function ProtectedRoute({ children, permission }) {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) return <Splash />;

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  if (permission && !can(user, permission)) {
    return <AccessRestricted user={user} permission={permission} />;
  }

  return children;
}
