use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use runnet::{Model, Tensor};

const MODEL_PATH: &str = "tests/fixtures/small_model.onnx";
const WARM_UP_RUNS: usize = 100;
const TIMED_RUNS: usize = 2000;

struct CountingAllocator;

static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn get_inputs() -> [Tensor; 3] {
    let wave = |count: usize, step: f32| {
        (0..count)
            .map(|at| (at as f32 * step).sin().abs())
            .collect::<Vec<_>>()
    };
    let targets: Vec<f32> = (0..5)
        .flat_map(|at| [150.0 + at as f32 * 40.0, 0.7 + at as f32 * 0.2])
        .collect();
    [
        Tensor::from_floats([5, 64], wave(5 * 64, 0.13)),
        Tensor::from_floats([5, 2], targets),
        Tensor::from_floats([3, 64], wave(3 * 64, 0.07)),
    ]
}

fn main() -> Result<(), runnet::Error> {
    let model = Model::load(MODEL_PATH)?;
    for _ in 0..WARM_UP_RUNS {
        model.run(get_inputs())?;
    }

    let inputs: Vec<[Tensor; 3]> = (0..TIMED_RUNS).map(|_| get_inputs()).collect();
    let first_count = ALLOCATION_COUNT.load(Ordering::Relaxed);
    let first_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let started = Instant::now();
    for run_inputs in inputs {
        model.run(run_inputs)?;
    }
    let microseconds = started.elapsed().as_secs_f64() * 1e6 / TIMED_RUNS as f64;
    let allocations = (ALLOCATION_COUNT.load(Ordering::Relaxed) - first_count) / TIMED_RUNS;
    let bytes = (ALLOCATED_BYTES.load(Ordering::Relaxed) - first_bytes) / TIMED_RUNS;

    println!(
        "{}: {:.1} us, {} allocations, {} bytes per run",
        MODEL_PATH, microseconds, allocations, bytes
    );
    Ok(())
}
