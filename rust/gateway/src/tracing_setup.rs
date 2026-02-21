use anyhow::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::config::Config;

/// Initialize tracing-subscriber with JSON output.
/// If OTel is enabled, also installs the OTLP span exporter.
pub fn init(config: &Config) -> Result<()> {
    let level = match config.log_level.to_uppercase().as_str() {
        "DEBUG" => "debug",
        "INFO" => "info",
        "WARN" | "WARNING" => "warn",
        "ERROR" => "error",
        _ => "info",
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));
    let json_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_current_span(false);

    if config.otel_enabled {
        match init_otel(config) {
            Ok(otel_layer) => {
                // OTel layer added first (closest to Registry), then fmt layers on top.
                tracing_subscriber::registry()
                    .with(otel_layer)
                    .with(filter)
                    .with(json_layer)
                    .init();
                tracing::info!(
                    endpoint = %config.otel_exporter_otlp_endpoint,
                    "OpenTelemetry OTLP exporter initialized"
                );
            }
            Err(e) => {
                // Fall back to JSON-only — write early warning to stderr.
                eprintln!(
                    "[gateway] OTel init failed ({}), continuing without distributed tracing",
                    e
                );
                tracing_subscriber::registry()
                    .with(filter)
                    .with(json_layer)
                    .init();
            }
        }
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(json_layer)
            .init();
    }

    Ok(())
}

fn init_otel(
    config: &Config,
) -> Result<impl tracing_subscriber::Layer<tracing_subscriber::Registry> + Send + Sync + 'static>
{
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::{runtime, trace as sdktrace, Resource};

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint(config.otel_exporter_otlp_endpoint.clone()),
        )
        .with_trace_config(
            sdktrace::Config::default().with_resource(Resource::new(vec![KeyValue::new(
                "service.name",
                "llm-gateway",
            )])),
        )
        .install_batch(runtime::Tokio)?;

    Ok(tracing_opentelemetry::layer().with_tracer(tracer))
}
