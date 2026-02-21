//! GPU Prometheus exporter — Rust replacement for `scripts/gpu_exporter.py`.
//!
//! Polls `nvidia-smi` every 2 seconds and exposes Prometheus metrics on
//! `GET /metrics`.  Also exposes `GET /health` for liveness checks.
//!
//! Configuration:
//!   GPU_EXPORTER_PORT — listen port (default: 9101)
use std::{sync::Arc, time::Duration};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use prometheus_client::{
    encoding::{text::encode, EncodeLabelSet},
    metrics::{family::Family, gauge::Gauge},
    registry::Registry,
};
use tokio::{process::Command, sync::Mutex, time::sleep};
use tracing::{error, info, warn};

// ── Prometheus label types ───────────────────────────────────────────────────

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct GpuLabels {
    gpu_id: String,
    gpu_name: String,
}

// ── Metric registry ───────────────────────────────────────────────────────────

struct GpuMetrics {
    registry: Registry,
    utilization: Family<GpuLabels, Gauge>,
    memory_used: Family<GpuLabels, Gauge>,
    memory_free: Family<GpuLabels, Gauge>,
    memory_total: Family<GpuLabels, Gauge>,
    memory_util: Family<GpuLabels, Gauge>,
    temperature: Family<GpuLabels, Gauge>,
    power_draw: Family<GpuLabels, Gauge>,
    power_limit: Family<GpuLabels, Gauge>,
    sm_clock: Family<GpuLabels, Gauge>,
    mem_clock: Family<GpuLabels, Gauge>,
    fan_speed: Family<GpuLabels, Gauge>,
    ecc_corrected: Family<GpuLabels, Gauge>,
    ecc_uncorrected: Family<GpuLabels, Gauge>,
}

impl GpuMetrics {
    fn new() -> Self {
        let mut registry = Registry::default();

        macro_rules! gauge_family {
            ($name:literal, $help:literal) => {{
                let f = Family::<GpuLabels, Gauge>::default();
                registry.register($name, $help, f.clone());
                f
            }};
        }

        let utilization = gauge_family!("gpu_utilization_percent", "GPU compute utilization (%)");
        let memory_used = gauge_family!("gpu_memory_used_mib", "GPU memory used (MiB)");
        let memory_free = gauge_family!("gpu_memory_free_mib", "GPU memory free (MiB)");
        let memory_total = gauge_family!("gpu_memory_total_mib", "GPU memory total (MiB)");
        let memory_util = gauge_family!(
            "gpu_memory_utilization_percent",
            "GPU memory bandwidth utilization (%)"
        );
        let temperature =
            gauge_family!("gpu_temperature_celsius", "GPU core temperature (°C)");
        let power_draw = gauge_family!("gpu_power_draw_watts", "GPU power draw (W)");
        let power_limit = gauge_family!("gpu_power_limit_watts", "GPU TDP power limit (W)");
        let sm_clock = gauge_family!("gpu_sm_clock_mhz", "GPU SM clock speed (MHz)");
        let mem_clock = gauge_family!("gpu_mem_clock_mhz", "GPU memory clock speed (MHz)");
        let fan_speed = gauge_family!("gpu_fan_speed_percent", "GPU fan speed (%)");
        let ecc_corrected = gauge_family!(
            "gpu_ecc_errors_corrected_total",
            "ECC corrected errors (volatile)"
        );
        let ecc_uncorrected = gauge_family!(
            "gpu_ecc_errors_uncorrected_total",
            "ECC uncorrected errors (volatile)"
        );

        GpuMetrics {
            registry,
            utilization,
            memory_used,
            memory_free,
            memory_total,
            memory_util,
            temperature,
            power_draw,
            power_limit,
            sm_clock,
            mem_clock,
            fan_speed,
            ecc_corrected,
            ecc_uncorrected,
        }
    }

    fn render(&self) -> String {
        let mut buf = String::new();
        encode(&mut buf, &self.registry).unwrap_or_default();
        buf
    }
}

// ── nvidia-smi polling ────────────────────────────────────────────────────────

const QUERY_FIELDS: &str = "index,name,\
    utilization.gpu,utilization.memory,\
    memory.used,memory.free,memory.total,\
    temperature.gpu,\
    power.draw,power.limit,\
    clocks.sm,clocks.mem,\
    fan.speed,\
    ecc.errors.corrected.volatile.total,\
    ecc.errors.uncorrected.volatile.total,\
    pcie.link.gen.current,pcie.link.width.current";

async fn collect_once(metrics: &GpuMetrics) {
    let output = Command::new("nvidia-smi")
        .args([
            &format!("--query-gpu={}", QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await;

    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            warn!(
                stderr = String::from_utf8_lossy(&o.stderr).as_ref(),
                "nvidia-smi exited with error"
            );
            return;
        }
        Err(e) => {
            error!(error = %e, "failed to run nvidia-smi");
            return;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.trim().lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 17 {
            continue;
        }

        let (idx, name) = (parts[0], parts[1]);
        let gpu_id = format!("gpu{}", idx);
        let labels = GpuLabels {
            gpu_id,
            gpu_name: name.to_string(),
        };

        macro_rules! set {
            ($metric:expr, $idx:expr) => {
                if let Ok(v) = safe_float(parts[$idx]) {
                    $metric.get_or_create(&labels).set(v);
                }
            };
        }

        set!(metrics.utilization, 2);
        set!(metrics.memory_util, 3);
        set!(metrics.memory_used, 4);
        set!(metrics.memory_free, 5);
        set!(metrics.memory_total, 6);
        set!(metrics.temperature, 7);
        set!(metrics.power_draw, 8);
        set!(metrics.power_limit, 9);
        set!(metrics.sm_clock, 10);
        set!(metrics.mem_clock, 11);
        set!(metrics.fan_speed, 12);
        set!(metrics.ecc_corrected, 13);
        set!(metrics.ecc_uncorrected, 14);
    }
}

fn safe_float(s: &str) -> Result<f64, ()> {
    s.split_whitespace()
        .next()
        .and_then(|tok| tok.parse::<f64>().ok())
        .ok_or(())
}

// ── App state ─────────────────────────────────────────────────────────────────

struct AppState {
    metrics: Arc<Mutex<GpuMetrics>>,
}

// ── Route handlers ────────────────────────────────────────────────────────────

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.metrics.lock().await.render();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let port: u16 = std::env::var("GPU_EXPORTER_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9101);

    let metrics = Arc::new(Mutex::new(GpuMetrics::new()));

    // Initial collection before the first scrape
    collect_once(&*metrics.lock().await).await;

    // Background collection loop (every 2 seconds)
    {
        let metrics = metrics.clone();
        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(2)).await;
                let guard = metrics.lock().await;
                collect_once(&*guard).await;
            }
        });
    }

    let state = Arc::new(AppState {
        metrics: metrics.clone(),
    });

    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    info!(%addr, "gpu-exporter listening");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
