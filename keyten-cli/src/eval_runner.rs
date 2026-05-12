//! Per-submission orchestration: parse, spawn a tokio current-thread
//! runtime, run `eval_async` with a progress monitor + Ctrl-C cancellation,
//! return the outcome.

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::task::LocalSet;

use keyten::{eval_async, parse, Ctx, Env, EvalErr, KernelErr, RefObj, RenderSink, RUNTIME};

use crate::progress::{run_progress_loop, ProgressPrinter};

pub enum Outcome {
    Ok(RefObj),
    Cancelled,
    Err(String),
}

const STRIDE: u64 = 1 << 20; // ~1M elements between progress wakes

pub fn run_one(text: &str, env: &mut Env) -> Result<Outcome> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .enable_io()
        .build()?;
    let local = LocalSet::new();
    let outcome = rt.block_on(local.run_until(async move {
        let cancelled = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(AtomicU64::new(0));
        let sink = Arc::new(RenderSink::with_stride(STRIDE));

        // Progress printer task.
        let progress_handle = tokio::task::spawn_local(run_progress_loop(
            sink.clone(),
            progress.clone(),
        ));

        // Build context; eval future borrows env mutably.
        let ctx = Ctx::new(&RUNTIME, &cancelled, &progress).with_render(&sink);

        let parsed = match parse(text) {
            Ok(e) => e,
            Err(e) => {
                progress_handle.abort();
                ProgressPrinter::clear();
                return Outcome::Err(format!("parse: {}", e.msg));
            }
        };

        let eval_fut = eval_async(&parsed, env, &ctx);

        let result = tokio::select! {
            r = eval_fut => map_outcome(r),
            _ = tokio::signal::ctrl_c() => {
                cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                // Wait a short grace period for the eval to observe cancellation
                // and tear down gracefully. After the timeout, we drop and move on.
                let _ = tokio::time::timeout(Duration::from_millis(250), async {
                    // Best-effort: the eval future was dropped by select!; this
                    // future is just to keep the runtime alive briefly so the
                    // progress task can flush. We use a sleep here.
                    tokio::time::sleep(Duration::from_millis(50)).await;
                })
                .await;
                Outcome::Cancelled
            }
        };

        progress_handle.abort();
        ProgressPrinter::clear();
        result
    }));
    Ok(outcome)
}

fn map_outcome(r: Result<RefObj, EvalErr>) -> Outcome {
    match r {
        Ok(v) => Outcome::Ok(v),
        Err(EvalErr::Kernel { err: KernelErr::Cancelled, .. }) => Outcome::Cancelled,
        Err(EvalErr::Kernel { err, .. }) => Outcome::Err(format!("kernel: {err:?}")),
        Err(EvalErr::UndefinedName { name, .. }) => {
            let s = decode_sym(name.0);
            Outcome::Err(format!("undefined name `{s}`"))
        }
        Err(EvalErr::Type { msg, .. }) => Outcome::Err(msg),
        Err(EvalErr::Empty) => Outcome::Err("empty expression".into()),
    }
}

fn decode_sym(packed: i64) -> String {
    packed
        .to_le_bytes()
        .iter()
        .copied()
        .take_while(|b| *b != 0)
        .map(|b| b as char)
        .collect()
}
