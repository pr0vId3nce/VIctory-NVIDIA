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
    /// Host copies this list brings up to date on the card
    /// ([`HostList::card_refresh`]): held until the list is reset, as a
    /// refresh step would hold them; and the ranges done (copy, address,
    /// bytes), each once until the list writes such memory.
    held: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
}

// SAFETY: D3D12 resources are free-threaded, and the copies are shared
// between threads as it is.
unsafe impl Send for CardScratch {}

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
        self.held.clear();
        self.refreshed.clear();
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
        self.refreshed.clear();
    }

    /// As [`CardScratch::written`], for a write of `len` bytes at `dst`:
    /// only one into such memory changes what a copy made here took (a
    /// list records thousands of buffer copies a frame, nearly all of them
    /// elsewhere).
    pub(super) fn written_at(&mut self, dst: usize, len: usize) {
        if !(self.made.is_empty() && self.refreshed.is_empty()) && crate::uma::touches(dst, len) {
            self.made.clear();
            self.refreshed.clear();
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

/// Since the last frame-rate line: refreshes of host copies done by a copy
/// on the card, their bytes, and those left to the CPU because shaders
/// write the copy or it has a side in the card's memory.
static REFRESHES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

impl HostList {
    /// A refresh of `[address, address + len)` of `shadow`, a host copy
    /// that is not in place, whose title bytes lie in a heap in the card's
    /// own memory: nothing watches that memory and a read of ours crosses
    /// the BAR, so the list copies the bytes itself, on the card, out of
    /// the buffer over the heap into the copy, here, where it is about to
    /// use them. `false`: not such a copy; the caller notes a refresh step
    /// as before.
    ///
    /// Not for a copy shaders write: its bytes from before are taken on the
    /// CPU at submission, when this copy has not run, and every byte it
    /// brings would count as written by them. Nor for one with a side in
    /// the card's memory, whose pieces' hashes are made where changes are
    /// sent up. Both stay on the CPU and show in the count of our reads.
    pub(crate) fn card_refresh(&self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) -> bool {
        if len == 0 || shadow.in_place() || !crate::uma::card_only(address, len) {
            return false;
        }
        if shadow.written() || shadow.carded() {
            REFRESHES[2].fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let Some((from, offset)) = crate::uma::source(address as u64, len as u64) else {
            return false;
        };
        let mut scratch = self.card_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if !scratch.refreshed.insert((std::sync::Arc::as_ptr(shadow) as usize, address, len)) {
            return true;
        }
        scratch.held.push(shadow.clone());
        drop(scratch);
        REFRESHES[0].fetch_add(1, Ordering::Relaxed);
        REFRESHES[1].fetch_add(len as u64, Ordering::Relaxed);
        // SAFETY: live buffers (the list holds the copy, the heap's buffer
        // stays for the run); the range lies in both.
        unsafe { self.list.CopyBufferRegion(&shadow.buffer, shadow.offset(address as u64), &from, offset, len as u64) };
        true
    }
}

/// For the frame-rate line.
pub(super) fn log(frames: f64) {
    let r: Vec<f64> = REFRESHES.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if r[0] + r[2] > 0.0 {
        tracing::info!(
            "UMA heap, host copies of bytes in card memory per frame: {:.1} refreshes done by a copy on the card in the list that reads them ({:.1} KB), {:.1} left              to the CPU (shaders write the copy, or it has a side on the card)",
            r[0],
            r[1] / 1024.0,
            r[2]
        );
    }
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
