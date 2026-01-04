# OpenTelemetry Observability

Complete guide for OpenTelemetry integration with Grafana stack (Alloy, Tempo, Prometheus, Loki).

## Overview

The template includes OpenTelemetry integration for traces, metrics, and logs. Designed for Grafana stack but works with any OTLP-compatible backend.

## Installation

```bash
# Install with telemetry support
pip install bm25-index-tool[telemetry]

# Or with uv
uv sync --extra telemetry
```

## Quick Start

### Enable Telemetry

```bash
# Via CLI flag
bm25-index-tool --telemetry

# Via environment variable
export OTEL_ENABLED=true
bm25-index-tool
```

### Development Mode

For development, use console exporter (default):

```bash
bm25-index-tool --telemetry -vv
```

This outputs traces, metrics, and logs to stderr.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `false` | Enable telemetry |
| `OTEL_SERVICE_NAME` | `bm25-index-tool` | Service name in traces |
| `OTEL_EXPORTER_TYPE` | `console` | `console` or `otlp` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | Alloy/Collector endpoint |
| `OTEL_EXPORTER_OTLP_INSECURE` | `true` | Use insecure connection |

### Production Setup (Grafana Alloy)

```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER_TYPE=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy:4317
bm25-index-tool
```

## Architecture (SOLID Design)

```
telemetry/
├── config.py      # TelemetryConfig - configuration from env vars (SRP)
├── service.py     # TelemetryService - singleton facade (ISP, DIP)
├── decorators.py  # @traced, trace_span - tracing utilities
└── exporters.py   # Exporter factory - extensible backends (OCP)
```

**Design Principles**:
- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Extensible via exporter factory
- **Liskov Substitution**: All exporters are interchangeable
- **Interface Segregation**: Clean, minimal interfaces
- **Dependency Inversion**: Depends on abstractions

## Usage Patterns

### 1. @traced Decorator

Automatically create spans for functions:

```python
from bm25_index_tool.telemetry import traced

@traced("process_data")
def process_data(items: list) -> dict:
    return {"count": len(items)}

@traced(attributes={"operation.type": "batch"})
def batch_process():
    pass
```

### 2. trace_span Context Manager

Manual span creation with custom attributes:

```python
from bm25_index_tool.telemetry import trace_span

with trace_span("database_query", {"db.system": "postgres"}) as span:
    result = db.execute(query)
    if span:
        span.set_attribute("db.rows", len(result))
```

### 3. Custom Metrics

Create custom counters and histograms:

```python
from bm25_index_tool.telemetry import TelemetryService

meter = TelemetryService.get_instance().meter

# Counter
counter = meter.create_counter("items_processed", description="Items processed")
counter.add(100, {"type": "batch"})

# Histogram
histogram = meter.create_histogram("processing_duration", unit="ms")
histogram.record(150.5, {"operation": "transform"})
```

## Local Observability Stack

A complete local observability stack is included in `references/`:

### Start the Stack

```bash
cd references
docker compose up -d
```

### Configure Your CLI

```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER_TYPE=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
bm25-index-tool -v
```

### Access Grafana

- URL: http://localhost:3000
- Username: `admin`
- Password: `admin`

### Stack Components

| Service | Port | Description |
|---------|------|-------------|
| **Grafana Alloy** | 4317, 4318 | OTLP receiver (gRPC, HTTP) |
| **Tempo** | 3200 | Distributed tracing backend |
| **Prometheus** | 9090 | Metrics storage and queries |
| **Loki** | 3100 | Log aggregation |
| **Grafana** | 3000 | Visualization and dashboards |

### Data Flow

```
bm25-index-tool --telemetry
      │ OTLP (gRPC:4317 or HTTP:4318)
      ▼
   Alloy ──► Tempo (traces)
      │
      ├────► Prometheus (metrics)
      │
      └────► Loki (logs)
              │
              ▼
           Grafana
```

## Stack Configuration

### Grafana Alloy

OTLP receiver that routes telemetry data to backends.

**Location**: `references/alloy/config.alloy`

**Key Features**:
- Receives OTLP over gRPC (4317) and HTTP (4318)
- Routes traces to Tempo
- Routes metrics to Prometheus
- Routes logs to Loki
- Batching and retry logic

### Tempo

Distributed tracing backend with S3-compatible storage.

**Location**: `references/tempo/tempo.yaml`

**Key Features**:
- Traces stored in `/tmp/tempo`
- OTLP ingestion on port 4317
- Query API on port 3200
- Automatic service graphs

### Prometheus

Metrics storage with PromQL query language.

**Location**: `references/prometheus/prometheus.yml`

**Key Features**:
- Scrapes Alloy metrics endpoint
- 15 day retention
- PromQL query interface
- Alert rules support

### Loki

Log aggregation with LogQL query language.

**Location**: `references/loki/loki.yaml`

**Key Features**:
- Logs stored in `/tmp/loki`
- LogQL query language
- Label-based indexing
- Integration with Grafana

### Grafana

Unified observability UI.

**Location**: `references/grafana/datasources.yml`

**Pre-configured Datasources**:
- Tempo (traces)
- Prometheus (metrics)
- Loki (logs)

**Features**:
- Trace visualization
- Metric dashboards
- Log exploration
- Service graphs
- Exemplars (traces linked to metrics)

## Example Queries

### Grafana Explore

**Traces (Tempo)**:
```
service.name="bm25-index-tool"
```

**Metrics (Prometheus)**:
```promql
rate(items_processed_total[5m])
histogram_quantile(0.95, rate(processing_duration_bucket[5m]))
```

**Logs (Loki)**:
```logql
{service_name="bm25-index-tool"} |= "error"
{service_name="bm25-index-tool"} | json | level="ERROR"
```

## Troubleshooting

### Telemetry Not Working

```bash
# Check if telemetry is enabled
echo $OTEL_ENABLED

# Try console exporter
bm25-index-tool --telemetry -vv

# Check Alloy logs
docker logs alloy

# Verify endpoint
curl http://localhost:4317
```

### No Traces in Tempo

```bash
# Check Tempo is running
curl http://localhost:3200/ready

# Check Alloy is forwarding
docker logs alloy | grep tempo

# Query Tempo API
curl http://localhost:3200/api/search
```

### No Metrics in Prometheus

```bash
# Check Prometheus targets
open http://localhost:9090/targets

# Check Alloy metrics endpoint
curl http://localhost:12345/metrics
```

## Performance Considerations

### Overhead

- **Console exporter**: Minimal (~1-2% overhead)
- **OTLP exporter**: Low (~2-5% overhead)
- **Batching**: Reduces network calls
- **Sampling**: Can be configured for high-volume services

### Production Recommendations

1. **Use OTLP exporter**: More efficient than console
2. **Enable batching**: Default in Alloy config
3. **Configure sampling**: For high-throughput services
4. **Monitor Alloy**: Ensure it's not a bottleneck
5. **Set retention**: Balance storage vs history needs

## Advanced Configuration

### Custom Exporters

Add new exporters in `telemetry/exporters.py`:

```python
def create_exporters(config: TelemetryConfig) -> dict[str, Any]:
    if config.exporter_type == "custom":
        return {
            "trace": CustomSpanExporter(),
            "metric": CustomMetricExporter(),
            "log": CustomLogExporter(),
        }
```

### Service Mesh Integration

For Kubernetes deployments with service mesh:

```yaml
# Inject OTEL_EXPORTER_OTLP_ENDPOINT via sidecar
env:
  - name: OTEL_ENABLED
    value: "true"
  - name: OTEL_EXPORTER_TYPE
    value: "otlp"
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://localhost:4317"
```

### Sampling

Configure sampling in `telemetry/service.py`:

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

# Sample 10% of traces
sampler = TraceIdRatioBased(0.1)
tracer_provider = TracerProvider(sampler=sampler)
```

## Resources

- **OpenTelemetry Docs**: https://opentelemetry.io/docs/
- **Grafana Alloy**: https://grafana.com/docs/alloy/latest/
- **Tempo Docs**: https://grafana.com/docs/tempo/latest/
- **Prometheus**: https://prometheus.io/docs/
- **Loki**: https://grafana.com/docs/loki/latest/
