//! Title memory in a heap the driver made, at the title's own address
//! (`VICTORY_UMA_HEAP`, off by default).
//!
//! Renderer port 1 (`crate::port`) and `crate::inplace` read title memory
//! in place through a heap *opened on* the title's pages
//! (`OpenExistingHeapFromAddress1`). On NVIDIA the card's loads and atomics
//! on such imported pages are about ten times slower than on a heap the
//! driver allocates itself (xcl-dev/xcl #2844's probe), which is why port 1
//! made the card slower on the RTX 4080 (19 to 32 ms). Here the pages are
//! the driver's instead: when the title maps physical pages into its 32-bit
//! graphics reservation, `xmem` lets go of the placeholder there and asks
//! [`title_memory_backed`] for a heap of that size whose CPU mapping lands
//! on the same address. Every other free stretch of the address space big
//! enough to take the mapping is reserved for a moment ("grooming", as
//! xcl's #2844 does), so the driver's `Map` can only land in the hole. The
//! title then writes the driver's memory and the card reads it; title
//! buffers and windows in it are buffers placed in that heap
//! ([`place_for`], through `crate::port::place_for`), with no copy, no
//! refresh, no write watch and nothing fetched back, as port 1's are.
//!
//! The heap is, by the switch:
//! - `upload`: `D3D12_HEAP_TYPE_GPU_UPLOAD`, the card's own memory written
//!   by the CPU through the resizable BAR (xcl #2938's mixed segments). The
//!   card reads it at its own speed; a CPU *read* crosses the bus uncached
//!   (12-84 MiB/s measured there).
//! - `custom`: a custom heap in the PC's memory, write-back, pool L0, with
//!   write watching (xcl #2844): the CPU reads it at full speed, the card
//!   over the bus, but as fast as a heap of the driver's.
//!
//! What it assumes is what port 1 assumes: what our CPU steps read there
//! at submission is what the title's CPU wrote. And: the title maps those
//! pages once and keeps them (port 1's notes: never tile-mapped, never
//! remapped). `xmem` warns if it does otherwise.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows_sys::Win32::System::Memory as mem;

use crate::inplace::Placed;

const K64: usize = 64 << 10;

/// What `VICTORY_UMA_HEAP` asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Off,
    Upload,
    Custom,
}

/// `VICTORY_UMA_HEAP`: `upload`, `custom`, or off (unset, anything else).
pub(crate) fn kind() -> Kind {
    static KIND: std::sync::OnceLock<Kind> = std::sync::OnceLock::new();
    *KIND.get_or_init(|| parse(std::env::var("VICTORY_UMA_HEAP").ok().as_deref()))
}

fn parse(value: Option<&str>) -> Kind {
    match value.map(str::trim) {
        Some("upload") => Kind::Upload,
        Some("custom") => Kind::Custom,
        _ => Kind::Off,
    }
}

/// Whether the switch is on (for `xmem`: read once).
pub fn uma_heap_on() -> bool {
    kind() != Kind::Off
}

/// A stretch of title memory that is a driver heap.
struct Backed {
    start: u64,
    end: u64,
    heap: ID3D12Heap,
    /// The buffer whose mapping the title's pages are: kept, mapped, for
    /// good (the mapping lives as long as it does).
    _buffer: ID3D12Resource,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Backed {}

static BACKED: Mutex<Vec<Backed>> = Mutex::new(Vec::new());

/// The lowest start and highest end of every stretch (both 0: none).
static BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Since the start: stretches backed and their bytes; asks where the
/// mapping landed elsewhere; asks the host refused; buffers and windows
/// placed in them; stretches groomed and the reservations that took.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// The title's `[base, base + len)` (64 KiB aligned, free: the caller let
/// go of the placeholder there) as a heap of the driver's, mapped at
/// `base`. `false`: nothing was made and the range is free again, for the
/// caller to reserve and map as before.
pub fn title_memory_backed(base: usize, len: usize) -> bool {
    let kind = kind();
    if kind == Kind::Off || base % K64 != 0 || len % K64 != 0 || len == 0 {
        return false;
    }
    let Some(host) = crate::host::host() else {
        return false;
    };
    if kind == Kind::Upload && !host.mapped_card_memory() {
        warn_once("VICTORY_UMA_HEAP=upload: this PC offers no GPU_UPLOAD heaps (resizable BAR); title memory stays as it was");
        return false;
    }
    let took = std::time::Instant::now();
    let groomed = groom(base, len);
    STATS[6].fetch_add(1, Ordering::Relaxed);
    STATS[7].fetch_add(groomed.len() as u64, Ordering::Relaxed);
    let made = make(&host.device, kind, len as u64);
    for at in &groomed {
        // SAFETY: our own reservations, nothing in them.
        unsafe { mem::VirtualFree(*at as *mut _, 0, mem::MEM_RELEASE) };
    }
    let (heap, buffer, data) = match made {
        Ok(made) => made,
        Err(e) => {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            tracing::warn!(base = format_args!("{base:#x}"), len, error = %e, ?kind, "UMA heap: the host refused the heap; title memory as before");
            return false;
        }
    };
    if data as usize != base {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            base = format_args!("{base:#x}"),
            len,
            landed = format_args!("{:#x}", data as usize),
            groomed = groomed.len(),
            ?kind,
            "UMA heap: the driver mapped the heap elsewhere; title memory as before"
        );
        // SAFETY: our own buffer, mapped once.
        unsafe { buffer.Unmap(0, None) };
        drop(buffer);
        drop(heap);
        return false;
    }
    crate::host::name(&buffer, || format!("title memory {base:#x}+{len:#x} in a driver heap"));
    let mut backed = BACKED.lock().unwrap_or_else(|e| e.into_inner());
    backed.push(Backed { start: base as u64, end: (base + len) as u64, heap, _buffer: buffer });
    let lo = backed.iter().map(|b| b.start).min().unwrap_or(0);
    let hi = backed.iter().map(|b| b.end).max().unwrap_or(0);
    BOUNDS[0].store(lo, Ordering::Relaxed);
    BOUNDS[1].store(hi, Ordering::Relaxed);
    drop(backed);
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(len as u64, Ordering::Relaxed);
    tracing::info!(
        base = format_args!("{base:#x}"),
        len,
        ?kind,
        groomed = groomed.len(),
        ms = took.elapsed().as_secs_f64() * 1000.0,
        "UMA heap: title memory is a driver heap at the title's address (no copy, no refresh, nothing fetched back)"
    );
    true
}

/// Whether `[at, at + len)` lies in a driver heap of ours.
pub fn title_memory_in_uma_heap(at: usize, len: usize) -> bool {
    let (at, end) = (at as u64, at as u64 + len.max(1) as u64);
    let hi = BOUNDS[1].load(Ordering::Relaxed);
    if hi == 0 || at >= hi || end <= BOUNDS[0].load(Ordering::Relaxed) {
        return false;
    }
    BACKED.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|b| b.start < end && at < b.end)
}

/// Title memory `[start, start + len)` in place, if a driver heap of ours
/// holds all of it: a buffer placed in that heap. `None`: not ours.
pub(crate) fn place_for(start: u64, len: u64) -> Option<Placed> {
    let hi = BOUNDS[1].load(Ordering::Relaxed);
    if hi == 0 || start >= hi {
        return None;
    }
    let at = start & !(K64 as u64 - 1);
    let end = (start + len.max(1)).next_multiple_of(K64 as u64);
    let heap = {
        let backed = BACKED.lock().unwrap_or_else(|e| e.into_inner());
        let b = backed.iter().find(|b| b.start <= at && end <= b.end)?;
        (b.heap.clone(), at - b.start)
    };
    let host = crate::host::host()?;
    let mut placed: Option<ID3D12Resource> = None;
    // SAFETY: plain creation in a live heap, inside it.
    let made = unsafe {
        host.device.CreatePlacedResource(
            &heap.0,
            heap.1,
            &buffer_desc(end - at),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut placed,
        )
    };
    match (made, placed) {
        (Ok(()), Some(buffer)) => {
            STATS[4].fetch_add(1, Ordering::Relaxed);
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
            Some(Placed { buffer, heap: heap.0, gpu, bias: start - at })
        }
        (made, _) => {
            if STATS[5].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(start = format_args!("{start:#x}"), len, error = ?made.err(), "UMA heap: no buffer placed in it; a copy as before");
            }
            None
        }
    }
}

/// For the frame-rate line.
pub(crate) fn log() {
    if kind() == Kind::Off {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    tracing::info!(
        "UMA heap (VICTORY_UMA_HEAP={:?}): {} stretches of title memory are driver heaps ({} MB); {} buffers and windows placed in them, {} not; \
         since the start {} mappings landed elsewhere, {} heaps refused; {} groomings, {} reservations",
        kind(),
        s[0],
        s[1] >> 20,
        s[4],
        s[5],
        s[2],
        s[3],
        s[6],
        s[7]
    );
}

fn warn_once(what: &str) {
    static SAID: AtomicU64 = AtomicU64::new(0);
    if SAID.fetch_add(1, Ordering::Relaxed) == 0 {
        tracing::warn!("{what}");
    }
}

fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

/// The heap, a buffer over all of it, and the buffer's CPU mapping.
fn make(device: &ID3D12Device, kind: Kind, size: u64) -> windows::core::Result<(ID3D12Heap, ID3D12Resource, *mut u8)> {
    let (properties, flags) = match kind {
        Kind::Upload => (D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
        _ => (
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH,
        ),
    };
    let desc = D3D12_HEAP_DESC { SizeInBytes: size, Properties: properties, Alignment: K64 as u64, Flags: flags };
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: plain creation.
    unsafe { device.CreateHeap(&desc, &mut heap)? };
    let heap = heap.ok_or_else(windows::core::Error::empty)?;
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation in the heap just made.
    unsafe { device.CreatePlacedResource(&heap, 0, &buffer_desc(size), D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)? };
    let buffer = buffer.ok_or_else(windows::core::Error::empty)?;
    let mut data = std::ptr::null_mut();
    // SAFETY: maps the whole buffer; nothing read (empty range).
    unsafe { buffer.Map(0, Some(&D3D12_RANGE::default()), Some(&mut data))? };
    Ok((heap, buffer, data.cast()))
}

/// Reserves every free stretch of the address space that a mapping of
/// `len` bytes could take, other than `[base, base + len)`, keeping a
/// little at the start of each for the driver's own small mappings: the
/// reservations' addresses, for the caller to release.
fn groom(base: usize, len: usize) -> Vec<usize> {
    // Less than the mapping, so that it cannot land there.
    let headroom = (len / 4).min(16 << 20).next_multiple_of(K64).min(len - K64.min(len));
    let mut out = Vec::new();
    let mut here = K64;
    // The highest user address on x64 Windows.
    const TOP: usize = 0x7fff_ffff_0000;
    while here < TOP {
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { mem::VirtualQuery(here as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let start = info.BaseAddress as usize;
        let end = start.saturating_add(info.RegionSize);
        if info.State == mem::MEM_FREE && !(start <= base && base + len <= end && end - start == len) {
            // Free stretches start on 64 KiB lines (the allocation grain).
            let from = (start + headroom).next_multiple_of(K64);
            let to = end & !(K64 - 1);
            if to > from && to - start >= len {
                // A free stretch that holds the hole and more: keep the
                // hole free, take what lies around it.
                for (a, b) in [(from, to.min(base)), (from.max(base + len), to)] {
                    if b > a && b - a >= K64 {
                        // SAFETY: reserves free address space, no pages.
                        let at = unsafe { mem::VirtualAlloc(a as *const _, b - a, mem::MEM_RESERVE, mem::PAGE_NOACCESS) };
                        if !at.is_null() {
                            out.push(at as usize);
                        }
                    }
                }
            }
        }
        if end <= here {
            break;
        }
        here = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_names_the_heap() {
        assert_eq!(parse(None), Kind::Off);
        assert_eq!(parse(Some("upload")), Kind::Upload);
        assert_eq!(parse(Some(" custom ")), Kind::Custom);
        assert_eq!(parse(Some("1")), Kind::Off);
    }

    /// The driver's mapping of a heap lands where the hole was, and the
    /// bytes written there can be written.
    /// Ignored by default: grooming takes the whole address space for a
    /// moment, which other tests' threads would trip over. Run alone:
    /// `cargo test -p victory-gpu uma -- --ignored --test-threads=1`.
    #[test]
    #[ignore]
    fn a_groomed_heap_lands_in_the_hole() {
        let Some(device) = crate::inplace::tests::warp() else { return };
        // Big enough that the headroom left in each stretch (a quarter, at
        // most 16 MiB) holds the device's own small allocations.
        let len = 64 << 20;
        // A hole of our own: reserve, note, release.
        // SAFETY: plain reserve and release of free address space.
        let base = unsafe { mem::VirtualAlloc(std::ptr::null(), len, mem::MEM_RESERVE, mem::PAGE_NOACCESS) } as usize;
        assert_ne!(base, 0);
        // SAFETY: our own reservation.
        unsafe { mem::VirtualFree(base as *mut _, 0, mem::MEM_RELEASE) };
        let groomed = groom(base, len);
        let made = make(&device, Kind::Custom, len as u64);
        for at in &groomed {
            // SAFETY: our own reservations.
            unsafe { mem::VirtualFree(*at as *mut _, 0, mem::MEM_RELEASE) };
        }
        let (_heap, buffer, data) = made.expect("a custom heap");
        assert_eq!(data as usize, base, "the mapping took the only hole big enough");
        // SAFETY: the mapping is `len` bytes.
        unsafe { std::ptr::write_bytes(data, 0x5a, len) };
        // SAFETY: our own buffer.
        unsafe { buffer.Unmap(0, None) };
    }
}
