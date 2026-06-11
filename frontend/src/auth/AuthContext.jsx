import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import api, { gicApi, TOKEN_KEY } from '../api/client';

const AuthContext = createContext(null);

function setAuthHeader(token) {
  if (token) {
    api.defaults.headers.common.Authorization = `Bearer ${token}`;
  } else {
    delete api.defaults.headers.common.Authorization;
  }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [loading, setLoading] = useState(true);

  // Hydrate the session on mount if a token is stored.
  useEffect(() => {
    let cancelled = false;
    const stored = localStorage.getItem(TOKEN_KEY);
    if (!stored) {
      setLoading(false);
      return;
    }
    setAuthHeader(stored);
    (async () => {
      try {
        const me = await gicApi.me();
        if (!cancelled) {
          setUser(me);
          setToken(stored);
        }
      } catch {
        if (!cancelled) {
          localStorage.removeItem(TOKEN_KEY);
          setAuthHeader(null);
          setUser(null);
          setToken(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const persistSession = useCallback((resp) => {
    const accessToken = resp.access_token;
    localStorage.setItem(TOKEN_KEY, accessToken);
    setAuthHeader(accessToken);
    setToken(accessToken);
    setUser(resp.user || null);
    return resp.user;
  }, []);

  const login = useCallback(
    async (username, password) => {
      const resp = await gicApi.login(username, password);
      return persistSession(resp);
    },
    [persistSession]
  );

  const loginWithDemo = useCallback(
    async (profile) => {
      const password = profile?.demo_password || profile?.password;
      const resp = await gicApi.login(profile.username, password);
      return persistSession(resp);
    },
    [persistSession]
  );

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setAuthHeader(null);
    setUser(null);
    setToken(null);
  }, []);

  const value = {
    user,
    token,
    loading,
    isAuthenticated: !!user,
    login,
    loginWithDemo,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return ctx;
}

export default AuthContext;
