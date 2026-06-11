import React from 'react';

export default function Loading({ message = 'Loading...', size = 'md' }) {
  const sizes = {
    sm: { spinner: 'w-5 h-5', text: 'text-xs' },
    md: { spinner: 'w-8 h-8', text: 'text-sm' },
    lg: { spinner: 'w-12 h-12', text: 'text-base' },
  };
  const { spinner, text } = sizes[size] || sizes.md;

  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12">
      <div
        className={`${spinner} rounded-full border-2 border-slate-600 border-t-blue-500 spin`}
        role="status"
        aria-label="Loading"
      />
      <span className={`${text} text-slate-400 font-medium`}>{message}</span>
    </div>
  );
}
