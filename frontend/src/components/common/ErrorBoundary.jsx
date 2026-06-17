import React from 'react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('[GIC ErrorBoundary]', error, info);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          className="rounded-xl p-8 border border-red-800 flex flex-col items-center gap-4 my-6"
          style={{ backgroundColor: '#1e293b' }}
        >
          <div className="text-3xl">⚠️</div>
          <div className="text-center">
            <h3 className="text-lg font-semibold text-red-400 mb-1">
              Something went wrong
            </h3>
            <p className="text-sm text-slate-400 max-w-md">
              {this.state.error?.message || 'An unexpected error occurred in this component.'}
            </p>
          </div>
          <button
            onClick={this.handleRetry}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors text-sm"
          >
            Retry
          </button>
          {process.env.NODE_ENV === 'development' && this.state.error && (
            <pre className="text-xs text-slate-500 text-left bg-slate-900 rounded p-3 max-w-full overflow-auto max-h-40 w-full">
              {this.state.error.stack}
            </pre>
          )}
        </div>
      );
    }
    return this.props.children;
  }
}
