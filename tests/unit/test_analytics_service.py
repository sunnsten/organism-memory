from __future__ import annotations

import pytest

from organism.shared.analytics import analytics
from organism.shared.analytics.service import AnalyticsService


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset Prometheus metrics between tests."""
    from prometheus_client import REGISTRY
    # Clear all collectors to avoid duplicates
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    # Clean up after test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def analytics_service():
    """Use global analytics singleton."""
    # Re-initialize metrics after clearing registry
    from organism.shared.analytics.service import AnalyticsService
    return AnalyticsService()


def test_analytics_service_metric_retrieval(analytics_service: AnalyticsService):
    """Test retrieval latency metric emission."""
    # Should not raise
    analytics_service.metric_retrieval(
        tenant_id="tenant1",
        latency_ms=0.09,
        tier="tier1",
    )

    # Verify metric exists in registry
    from prometheus_client import REGISTRY
    metrics = REGISTRY.get_sample_value(
        "retrieval_latency_ms_count",
        {"tenant": "tenant1", "method": "hnsw", "tier": "tier1"},
    )
    assert metrics == 1.0


def test_analytics_service_metric_consolidation(analytics_service: AnalyticsService):
    """Test consolidation rate metric emission."""
    analytics_service.metric_consolidation(
        tenant_id="tenant1",
        promoted=5,
        total=10,
    )

    from prometheus_client import REGISTRY
    counter = REGISTRY.get_sample_value(
        "consolidation_events_total",
        {"tenant": "tenant1", "result": "promoted"},
    )
    assert counter == 5.0


def test_analytics_service_metric_write(analytics_service: AnalyticsService):
    """Test write event metric emission."""
    analytics_service.metric_write(
        tenant_id="tenant1",
        importance=0.8,
        filtered=False,
    )

    from prometheus_client import REGISTRY
    counter = REGISTRY.get_sample_value(
        "write_events_total",
        {"tenant": "tenant1", "filtered": "false"},
    )
    assert counter == 1.0


def test_analytics_service_emit_event(analytics_service: AnalyticsService, caplog):
    """Test structured event emission (logging)."""
    import logging
    caplog.set_level(logging.INFO)

    analytics_service.emit_event(
        event_type="retrieval.completed",
        tenant_id="tenant1",
        latency_ms=0.09,
        tier="tier1",
    )

    # Verify log was emitted (event_type in message, tenant_id in extra)
    assert "retrieval.completed" in caplog.text
    # Check the log record's extra fields
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.tenant_id == "tenant1"
    assert record.event_type == "retrieval.completed"


def test_analytics_service_tenant_isolation(analytics_service: AnalyticsService):
    """Test that metrics are isolated per tenant."""
    # Emit for tenant1
    analytics_service.metric_retrieval("tenant1", 0.09, "tier1")
    # Emit for tenant2
    analytics_service.metric_retrieval("tenant2", 50.0, "tier1")

    from prometheus_client import REGISTRY
    t1_count = REGISTRY.get_sample_value(
        "retrieval_latency_ms_count",
        {"tenant": "tenant1", "method": "hnsw", "tier": "tier1"},
    )
    t2_count = REGISTRY.get_sample_value(
        "retrieval_latency_ms_count",
        {"tenant": "tenant2", "method": "hnsw", "tier": "tier1"},
    )

    assert t1_count == 1.0
    assert t2_count == 1.0  # Separate counters
