from __future__ import annotations

import logging
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Shared analytics service for metrics and observability.

    Provides:
    - Prometheus metrics (retrieval latency, consolidation rate, etc.)
    - Structured event logging
    - Tenant isolation via labels
    - SaaS-ready monitoring

    Architecture: Cross-cutting concern, used by all layers (Core/Research/Agents).
    """

    def __init__(self):
        """Initialize metrics registry."""
        # Retrieval latency (vectorlite HNSW performance)
        self.retrieval_latency = Histogram(
            "retrieval_latency_ms",
            "Retrieval latency in milliseconds (vectorlite HNSW)",
            labelnames=["tenant", "method", "tier"],
            buckets=[0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500],  # ms
        )

        # Consolidation events
        self.consolidation_rate = Counter(
            "consolidation_events_total",
            "Total consolidation events",
            labelnames=["tenant", "result"],  # result: promoted/rejected/merged
        )

        # Token savings gauge
        self.token_savings = Gauge(
            "token_savings_ratio",
            "Token savings ratio vs full history",
            labelnames=["tenant"],
        )

        # Write events
        self.write_events = Counter(
            "write_events_total",
            "Total write events (EventRecord → ExperienceBlock)",
            labelnames=["tenant", "filtered"],  # filtered: true/false
        )

        # Active tenants gauge
        self.active_tenants = Gauge(
            "active_tenants_total",
            "Number of active tenants",
        )

        # HTTP request metrics
        self.http_requests = Counter(
            "http_requests_total",
            "Total HTTP requests",
            labelnames=["method", "endpoint", "status"],
        )
        self.http_request_duration = Histogram(
            "http_request_duration_ms",
            "HTTP request duration in milliseconds",
            labelnames=["method", "endpoint"],
            buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 5000, 30000],
        )

    def metric_http(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration_ms: float,
    ) -> None:
        self.http_requests.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration_ms)

    def emit_event(self, event_type: str, tenant_id: str, **kwargs: Any) -> None:
        """
        Emit structured event (Prometheus + JSONL log).

        Args:
            event_type: Event type (e.g., "retrieval.completed")
            tenant_id: Tenant identifier
            **kwargs: Additional event data
        """
        logger.info(
            event_type,
            extra={
                "tenant_id": tenant_id,
                "event_type": event_type,
                **kwargs,
            },
        )

    def metric_retrieval(
        self,
        tenant_id: str,
        latency_ms: float,
        tier: str,
        method: str = "hnsw",
    ) -> None:
        """
        Record retrieval latency metric.

        Args:
            tenant_id: Tenant identifier
            latency_ms: Latency in milliseconds
            tier: Tier name (tier0/tier1/tier2)
            method: Retrieval method (hnsw/python/fts)
        """
        self.retrieval_latency.labels(
            tenant=tenant_id,
            method=method,
            tier=tier,
        ).observe(latency_ms)

    def metric_consolidation(
        self,
        tenant_id: str,
        promoted: int,
        total: int,
    ) -> None:
        """
        Record consolidation metrics.

        Args:
            tenant_id: Tenant identifier
            promoted: Number of candidates promoted to memory items
            total: Total candidates processed
        """
        self.consolidation_rate.labels(
            tenant=tenant_id,
            result="promoted",
        ).inc(promoted)

        # Calculate promotion rate for dashboard
        rate = promoted / total if total > 0 else 0.0
        logger.debug(
            "Consolidation rate: %.2f%% (%d/%d)",
            rate * 100,
            promoted,
            total,
        )

    def metric_write(
        self,
        tenant_id: str,
        importance: float,
        filtered: bool,
    ) -> None:
        """
        Record write event metric.

        Args:
            tenant_id: Tenant identifier
            importance: Event importance score
            filtered: Whether event was filtered (below threshold)
        """
        self.write_events.labels(
            tenant=tenant_id,
            filtered=str(filtered).lower(),
        ).inc()

    def metric_token_savings(
        self,
        tenant_id: str,
        saved_tokens: int,
        total_tokens: int,
    ) -> None:
        """
        Record token savings metric.

        Args:
            tenant_id: Tenant identifier
            saved_tokens: Tokens saved by using memory vs full history
            total_tokens: Total tokens if full history was used
        """
        ratio = saved_tokens / total_tokens if total_tokens > 0 else 0.0
        self.token_savings.labels(tenant=tenant_id).set(ratio)


# Global singleton instance
analytics = AnalyticsService()

__all__ = ["AnalyticsService", "analytics"]
