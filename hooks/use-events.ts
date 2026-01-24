"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useEventsStore } from "@/stores/events-store";
import { useAuthStore } from "@/stores/auth-store";
import type { ThreatEvent } from "@/types";

const APP_MODE = process.env.NEXT_PUBLIC_APP_MODE || "self-hosted";

interface UseEventsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  queries?: string[];
}

export function useEvents(options: UseEventsOptions = {}) {
  const {
    autoRefresh = true,
    refreshInterval = 60000,
    queries,
  } = options;

  const {
    events,
    filteredEvents,
    isLoading,
    error,
    setEvents,
    addEvents,
    setLoading,
    setError,
  } = useEventsStore();

  const { getAccessToken, signOut, isAuthenticated } = useAuthStore();
  const [requiresSignIn, setRequiresSignIn] = useState(false);

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const initialFetchRef = useRef(false);
  const eventsRef = useRef(events);

  // Keep events ref updated
  useEffect(() => {
    eventsRef.current = events;
  }, [events]);

  const requiresAuth = APP_MODE === "valyu";

  const fetchEvents = useCallback(async (isInitialLoad = false) => {
    // After initial load, require sign-in for refreshes (in valyu mode only)
    if (requiresAuth && !isInitialLoad && !isAuthenticated) {
      setRequiresSignIn(true);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const accessToken = getAccessToken();

      const response = await fetch("/api/events", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ queries: queries || [], accessToken }),
      });

      const data = await response.json();

      // Handle auth errors (401 or requiresReauth flag)
      if (!response.ok || data.requiresReauth) {
        if (response.status === 401 || data.requiresReauth) {
          signOut();
          setRequiresSignIn(true);
          setError("Session expired. Please sign in again.");
          return;
        }
        throw new Error(data.error || "Failed to fetch events");
      }

      const newEvents: ThreatEvent[] = data.events || [];

      if (!initialFetchRef.current) {
        setEvents(newEvents);
        initialFetchRef.current = true;
      } else {
        const existingIds = new Set(eventsRef.current.map((e) => e.id));
        const trulyNewEvents = newEvents.filter((e) => !existingIds.has(e.id));

        if (trulyNewEvents.length > 0) {
          addEvents(trulyNewEvents);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setLoading(false);
    }
  }, [queries, setEvents, addEvents, setLoading, setError, getAccessToken, signOut, requiresAuth, isAuthenticated]);

  const refresh = useCallback(() => {
    // After initial load, require sign-in for refreshes
    if (requiresAuth && !isAuthenticated) {
      setRequiresSignIn(true);
      return;
    }
    fetchEvents(false);
  }, [fetchEvents, requiresAuth, isAuthenticated]);

  // Initial fetch - runs once on mount
  useEffect(() => {
    if (!initialFetchRef.current) {
      fetchEvents(true);
    }
  }, []); // Empty deps - only run on mount

  // Auto-refresh - only if authenticated
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (autoRefresh && isAuthenticated && initialFetchRef.current) {
      intervalRef.current = setInterval(() => fetchEvents(false), refreshInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoRefresh, refreshInterval, isAuthenticated, fetchEvents]);

  useEffect(() => {
    if (isAuthenticated) {
      setRequiresSignIn(false);
    }
  }, [isAuthenticated]);

  return {
    events,
    filteredEvents,
    isLoading,
    error,
    refresh,
    requiresSignIn,
  };
}
