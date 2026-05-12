//! Verb kernels. Each module exposes step structs (ChunkStep impls) and the
//! per-kind-pair dispatch entry points used by `op::dispatch_*`.

pub mod plus;
pub mod minus;
pub mod times;
pub mod div;
pub mod til;
pub mod monad;
pub mod dyad;
pub mod compare;
pub mod minmax;

// Per-kind streaming chunk defaults. Sized so x_chunk + y_chunk + out_chunk
// fit comfortably in L2 cache (~512 KiB working set across the three buffers).
pub const I64_CHUNK: usize = 64 * 1024; //  64K i64 = 512 KB per buffer
pub const F64_CHUNK: usize = 64 * 1024;
pub const I32_CHUNK: usize = 128 * 1024;
pub const F32_CHUNK: usize = 128 * 1024;
pub const I16_CHUNK: usize = 256 * 1024;
pub const U8_CHUNK: usize = 512 * 1024;
pub const BOOL_CHUNK: usize = 512 * 1024;
