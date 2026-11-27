//! Constant buffers out of title memory in the card's own memory
//! (`VICTORY_UMA_HEAP=upload`, `crate::uma`), copied by the card.
//!
//! A table's constant buffer is copied afresh for every draw that names it:
//! the console takes one on any 16 bytes, D3D12 on 256, so its bytes go
//! into upload memory of the list's on 256 (`Step::Fill`, at submission, on
//! the CPU). Out of a heap in the card's memory that read crosses the BAR
//! at about 25 MiB/s: 9,600 of them a frame were 183 ms. Translated shaders
//! could read the title's bytes where they are through a raw view
//! (`crate::port_cbv`), but that takes a twin of the slot, which a view
//! heap of 1,000,000 has no room for.
//!
//! So here the list copies them itself, on the card, right where the draw
//! is recorded: out of the buffer over the heap (`crate::uma::source`) into
//! card memory of the list's own on 256 bytes, and the slot shows that.
//! Nothing is left for the submission: no step, no fill, no read of ours.
//! One copy for every place and size a list names, however many of its
//! tables name it.
//!
//! What the card copies is what the heap holds when the list runs: the
//! title's bytes as of then, as for every buffer in place (renderer port
//! 1's rule; the title writes its constants into a ring it waits on its
//! fences for).
//!
//! shortcut: the chunk is in the common state and is given no barriers
//! between its copies and the draws that read it; on a buffer the drivers
//! ask for none, the debug layer would.

use super::*;

/// Card memory handed out front to back, on 256 bytes: the chunks, the one
/// in use and the bytes used in it; and where each place copied so far
/// lies (title address, bytes: chunk, offset).
#[derive(Default)]
pub(crate) struct CardScratch {
    chunks: Vec<ID3D12Resource>,
    current: usize,
    used: u64,
    made: std::collections::HashMap<(u64, u32), (usize, u64)>,
}

/// A chunk's bytes.
const CHUNK: u64 = 1 << 20;

/// Chunks no list holds, and the most kept.
struct Spare(Vec<ID3D12Resource>);

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Spare {}

static SPARE: Mutex<Spare> = Mutex::new(Spare(Vec::new()));
const SPARE_MOST: usize = 64;

/// Since the last frame-rate line: constant buffers asked for, copies
/// recorded, their bytes, and asks turned down (too big, no chunk).
static STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

impl CardScratch {
    /// The list was reset: its chunks are the next list's.
    pub(super) fn reset(&mut self) {
        self.made.clear();
        (self.current, self.used) = (0, 0);
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for chunk in self.chunks.drain(..) {
            if spare.0.len() < SPARE_MOST {
                spare.0.push(chunk);
            }
        }
    }

    /// The copies made so far are not what a later ask means: the list
    /// writes title memory in between.
    pub(super) fn written(&mut self) {
        self.made.clear();
    }

    /// As [`CardScratch::written`], for a write of `len` bytes at `dst`:
    /// only one into such memory changes what a copy made here took (a
    /// list records thousands of buffer copies a frame, nearly all of them
    /// elsewhere).
    pub(super) fn written_at(&mut self, dst: usize, len: usize) {
        if !self.made.is_empty() && crate::uma::touches(dst, len) {
            self.made.clear();
        }
    }

    /// Room for `len` bytes (a multiple of 256, at most [`CHUNK`]): the
    /// chunk's number and the offset in it.
    fn room(&mut self, host: &Host, len: u64) -> Option<(usize, u64)> {
        if self.current < self.chunks.len() && self.used + len > CHUNK {
            (self.current, self.used) = (self.current + 1, 0);
        }
        if self.current == self.chunks.len() {
            let spare = SPARE.lock().unwrap_or_else(|e| e.into_inner()).0.pop();
            let chunk = match spare {
                Some(chunk) => chunk,
                None => {
                    let chunk = host.uav_buffer(CHUNK)?;
                    name(&chunk, || "list constants on the card".into());
                    chunk
                }
            };
            self.chunks.push(chunk);
            self.used = 0;
        }
        let at = self.used;
        self.used += len;
        Some((self.current, at))
    }
}

impl HostList {
    /// The constant buffer of `size` bytes at title address `address`, if
    /// its bytes lie in a heap in the card's memory: copied by this list,
    /// on the card, at this place in it, into card memory of its own. The
    /// buffer (for a slot to hold) and the host address, on 256 bytes.
    /// `None`: not such memory, or no room; the caller does what it did.
    pub(crate) fn card_constants(&self, address: u64, size: u32) -> Option<(ID3D12Resource, u64)> {
        if size == 0 || !crate::uma::card_only(address as usize, size as usize) {
            return None;
        }
        STATS[0].fetch_add(1, Ordering::Relaxed);
        let len = u64::from(size).next_multiple_of(256);
        let found = (len <= CHUNK).then(|| crate::uma::source(address, u64::from(size))).flatten();
        let (Some((from, offset)), Some(host)) = (found, host()) else {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let mut scratch = self.card_scratch.lock().unwrap_or_else(|e| e.into_inner());
        let (chunk, at, new) = match scratch.made.get(&(address, size)) {
            Some(&(chunk, at)) => (chunk, at, false),
            None => {
                let Some((chunk, at)) = scratch.room(&host, len) else {
                    STATS[3].fetch_add(1, Ordering::Relaxed);
                    return None;
                };
                scratch.made.insert((address, size), (chunk, at));
                (chunk, at, true)
            }
        };
        let buffer = scratch.chunks[chunk].clone();
        drop(scratch);
        if new {
            STATS[1].fetch_add(1, Ordering::Relaxed);
            STATS[2].fetch_add(u64::from(size), Ordering::Relaxed);
            // SAFETY: both buffers are alive (the list's chunk, the heap's
            // own buffer) and the ranges lie inside them.
            unsafe { self.list.CopyBufferRegion(&buffer, at, &from, offset, u64::from(size)) };
        }
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() } + at;
        Some((buffer, gpu))
    }
}

/// For the frame-rate line.
pub(super) fn log(frames: f64) {
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if s[0] == 0.0 {
        return;
    }
    tracing::info!(
        "UMA heap, constant buffers out of card memory per frame: {:.0} asked for, {:.0} copies recorded by the lists that read them ({:.1} KB; no step at \
         submission, nothing read on the CPU), {:.1} turned down (copied on the CPU as before)",
        s[0],
        s[1],
        s[2] / 1024.0,
        s[3]
    );
}
