//! Profiling harness for the Rust port.  Reads the shared dataset + model
//! written by `parity/profile.py` and times the pipeline phases, reporting
//! peak resident memory.
//!
//! Run with: `cargo run --release --example profile [cores] [--json]`

use std::path::PathBuf;
use std::time::Instant;

use dedupe::api::{pairs_dedupe_ids, Settings};
use dedupe::clustering;
use dedupe::value::{Data, Record, RecordId, Value};
use serde_json::Value as J;

/// Peak resident set size in bytes.
fn peak_rss_bytes() -> u64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(
                process: isize,
                counters: *mut ProcessMemoryCounters,
                cb: u32,
            ) -> i32;
        }
        // SAFETY: standard Win32 query with a correctly-sized, zeroed struct.
        unsafe {
            let mut c: ProcessMemoryCounters = std::mem::zeroed();
            c.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
            if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0 {
                return c.peak_working_set_size as u64;
            }
        }
        0
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    if let Some(kb) = rest.split_whitespace().next() {
                        if let Ok(kb) = kb.parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
        0
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        0
    }
}

fn decode_plain(j: &J) -> Value {
    match j {
        J::Null => Value::Null,
        J::Bool(b) => Value::Bool(*b),
        J::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        J::String(s) => Value::Str(s.clone()),
        J::Array(items) => Value::List(items.iter().map(decode_plain).collect()),
        _ => Value::Null,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_mode = args.iter().any(|a| a == "--json");
    let cores: usize = args
        .iter()
        .skip(1)
        .find_map(|a| a.parse().ok())
        .unwrap_or(1);

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("profile.json");
    let text =
        std::fs::read_to_string(&path).expect("read profile.json (run `python profile.py gen`)");
    let golden: J = serde_json::from_str(&text).expect("parse profile.json");

    let settings: Settings =
        serde_json::from_value(golden["settings"].clone()).expect("deserialize settings");
    let matching = settings.to_matching(cores, true).expect("build matching");

    let mut data = Data::new();
    for (k, v) in golden["data"].as_object().unwrap() {
        let id = RecordId::Int(k.parse().unwrap());
        let mut record = Record::new();
        if let Some(obj) = v.as_object() {
            for (field, value) in obj {
                record.insert(field.clone(), decode_plain(value));
            }
        }
        data.insert(id, record);
    }

    let fp = matching.fingerprinter().unwrap().clone_ref();

    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);

    let t0 = Instant::now();
    fp.index_all(&data);
    let t1 = Instant::now();
    let rss1 = peak_rss_bytes();
    let pairs = pairs_dedupe_ids(&fp, &data);
    let t2 = Instant::now();
    let rss2 = peak_rss_bytes();
    let scores = matching.score_ids(&pairs, &data);
    let t3 = Instant::now();
    let rss3 = peak_rss_bytes();
    let clusters = clustering::cluster(&scores, 0.5, 30_000);
    let t4 = Instant::now();
    let rss4 = peak_rss_bytes();

    let index_all = (t1 - t0).as_secs_f64();
    let blocking = (t2 - t1).as_secs_f64();
    let score = (t3 - t2).as_secs_f64();
    let cluster = (t4 - t3).as_secs_f64();
    let total = (t4 - t0).as_secs_f64();
    let peak_mb = mb(rss4);
    let m = |b: u64| format!("{:.0}", mb(b));

    if json_mode {
        println!(
            "{{\"impl\":\"rust\",\"cores\":{cores},\"records\":{},\"pairs\":{},\"scored\":{},\"clusters\":{},\
             \"index_all\":{index_all:.4},\"blocking\":{blocking:.4},\"score\":{score:.4},\
             \"cluster\":{cluster:.4},\"total\":{total:.4},\"peak_mb\":{peak_mb:.2},             \"peak_index_mb\":{},\"peak_pairs_mb\":{},\"peak_score_mb\":{},\"peak_cluster_mb\":{}}}",
            data.len(),
            pairs.len(),
            scores.len(),
            clusters.len(),
            m(rss1),
            m(rss2),
            m(rss3),
            m(rss4),
        );
    } else {
        println!("Rust pipeline ({} records, {} core(s)):", data.len(), cores);
        println!("  index_all      : {index_all:8.3}s");
        println!("  blocking+pairs : {blocking:8.3}s ({} pairs)", pairs.len());
        println!("  score          : {score:8.3}s ({} scored)", scores.len());
        println!("  cluster        : {cluster:8.3}s ({} clusters)", clusters.len());
        println!("  TOTAL          : {total:8.3}s");
        println!("  peak RSS       : {peak_mb:8.1} MB");
    }
}
