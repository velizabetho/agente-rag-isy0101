"""Export turn records to OpenTelemetry: traces (Tempo) and metrics (Prometheus), shown in Grafana.

Enabled only when OTEL_EXPORTER_OTLP_ENDPOINT is set (e.g. http://localhost:4318 with the grafana/otel-lgtm
container from observability/grafana). The same record that is written to the JSONL log is turned into a span
tree with the original timings, so Grafana and the Streamlit dashboard show the same data.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from bookclub.config import ROOT_DIR

SERVICE_NAME = "agente-clubes-lectura"
SECONDS_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30, 45, 60, 90, 120, 180]


class _Instruments:
    def __init__(self, meter: metrics.Meter):
        self.turn = meter.create_histogram("agent.turn.duration", unit="s", description="End-to-end latency of a turn",
                                           explicit_bucket_boundaries_advisory=SECONDS_BUCKETS)
        self.step = meter.create_histogram("agent.step.duration", unit="s",
                                           description="Latency of nodes, LLM calls, tools and retrieval steps",
                                           explicit_bucket_boundaries_advisory=SECONDS_BUCKETS)
        self.tokens = meter.create_counter("agent.tokens", unit="{token}", description="LLM tokens by model and type")
        self.cost = meter.create_counter("agent.cost", unit="usd", description="Estimated LLM cost")
        self.turns = meter.create_counter("agent.turns", description="Turns by status")
        self.errors = meter.create_counter("agent.errors", description="Errors by type and source")
        self.tool_calls = meter.create_counter("agent.tool.calls", description="Tool calls by tool and status")
        self.guardrail = meter.create_counter("agent.guardrail.events", description="Guardrail blocks and masking")
        self.retry_wait = meter.create_counter("agent.provider.retry_wait", unit="s",
                                               description="Time waiting for provider retries (HTTP 429)")
        self.rss = meter.create_gauge("agent.process.rss", unit="MiBy", description="Resident memory after the turn")
        self.cpu = meter.create_histogram("agent.turn.cpu", unit="s", description="CPU time used by the turn",
                                          explicit_bucket_boundaries_advisory=SECONDS_BUCKETS)


@lru_cache
def _setup() -> tuple[trace.Tracer, _Instruments] | None:
    load_dotenv(ROOT_DIR / ".env")  # OTEL_* variables are read by the exporters from the environment
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", SERVICE_NAME)})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=5000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    return tracer_provider.get_tracer(SERVICE_NAME), _Instruments(meter_provider.get_meter(SERVICE_NAME))


def enabled() -> bool:
    return _setup() is not None


def _parent_of(item: dict, candidates: list[dict]) -> dict | None:
    """Innermost candidate (tool before node) whose time window contains the item."""
    start = item["start_ms"]
    for kind in ("tool", "node"):
        for c in candidates:
            if c is not item and c["kind"] == kind and c["start_ms"] <= start <= c["start_ms"] + c["duration_ms"]:
                if kind == "node" and item["kind"] == "node":
                    continue
                return c
    return None


def export_turn(record: dict, start_ns: int) -> None:
    setup = _setup()
    if setup is None:
        return
    tracer, inst = setup
    labels = {"category": record.get("category") or "app", "model": record.get("agent_model") or "unknown"}

    root_attrs = {
        "session.id": record.get("thread_id", ""), "club.id": record.get("club_id", ""),
        "agent.status": record["status"], "agent.source": record.get("source", ""),
        "agent.scenario": record.get("scenario_id") or "", "agent.tokens.total": record["total_tokens"],
        "agent.llm_calls": record["llm_calls"], "agent.tool_calls": record["tool_calls"],
        "agent.cost_usd": record["cost_usd"], "agent.http_429": record["http_429"],
        "agent.retry_wait_s": record["retry_wait_s"], "agent.guardrail": str(record.get("guardrail") or ""),
        "input.value": record.get("input", ""), "output.value": record.get("answer", ""),
    }
    end_ns = start_ns + int(record["latency_ms"] * 1e6)
    root = tracer.start_span("agent.turn", start_time=start_ns, attributes=root_attrs)
    if record["status"] == "error":
        root.set_status(Status(StatusCode.ERROR))
    otel_spans: dict[int, trace.Span] = {}
    for item in record["spans"]:  # sorted by start, so parents are created before children
        parent = _parent_of(item, record["spans"])
        parent_span = otel_spans.get(id(parent)) if parent else root
        attrs = {"agent.kind": item["kind"], "agent.node": item.get("node") or ""}
        if item["kind"] == "llm":
            attrs.update({"gen_ai.request.model": item["model"], "gen_ai.usage.input_tokens": item.get("input_tokens", 0),
                          "gen_ai.usage.output_tokens": item.get("output_tokens", 0)})
        if item["kind"] == "tool":
            attrs["gen_ai.tool.name"] = item["name"]
        s = tracer.start_span(item["name"], context=trace.set_span_in_context(parent_span),
                              start_time=start_ns + int(item["start_ms"] * 1e6), attributes=attrs)
        if item["status"] == "error":
            s.set_status(Status(StatusCode.ERROR, item.get("error", "")[:200]))
            s.set_attribute("error.type", item.get("error_type", "other"))
        s.end(end_time=start_ns + int((item["start_ms"] + item["duration_ms"]) * 1e6))
        otel_spans[id(item)] = s

        step_labels = {"kind": item["kind"], "name": item["name"], "status": item["status"]}
        inst.step.record(item["duration_ms"] / 1000, step_labels)
        if item["kind"] == "llm":
            for kind in ("input", "output", "cached"):
                inst.tokens.add(item.get(f"{kind}_tokens", 0), {"model": item["model"], "type": kind})
        if item["kind"] == "tool":
            inst.tool_calls.add(1, {"tool": item["name"], "status": item["status"]})
    root.end(end_time=end_ns)

    inst.turn.record(record["latency_ms"] / 1000, {**labels, "status": record["status"]})
    inst.turns.add(1, {**labels, "status": record["status"]})
    inst.cost.add(record["cost_usd"], labels)
    inst.cpu.record(record["cpu_ms"] / 1000, labels)
    inst.rss.set(record["rss_mb"])
    inst.retry_wait.add(record["retry_wait_s"], labels)
    for err in record["errors"]:
        inst.errors.add(1, {"type": err["type"], "source": err["source"]})
    if record.get("failure") and not record["errors"]:
        inst.errors.add(1, {"type": record["failure"]["type"], "source": "turn"})
    if record["http_429"]:
        inst.errors.add(record["http_429"], {"type": "http_429", "source": "provider"})
    guard = record.get("guardrail") or {}
    if guard.get("reason"):
        inst.guardrail.add(1, {"event": guard["reason"]})
    for kind in guard.get("pii", []):
        inst.guardrail.add(1, {"event": f"pii_{kind}"})
