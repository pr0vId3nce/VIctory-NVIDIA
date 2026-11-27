//! Title memory in a heap the driver made, at the title's own address
//! (`VICTORY_UMA_HEAP`, off by default).
//!
//! Renderer port 1 (`crate::port`) and `crate::inplace` read title memory
//! in place through a heap *opened on* the title's pages
//! (`OpenExistingHeapFromAddress1`). On NVIDIA the card's loads and atomics
//! on such imported pages are about ten times slower than on a heap the
//! driver allocates itself (xcl-dev/xcl #2844's probe), which is why port 1
//! made the card slower on the RTX 4080 (19 to 32 ms). Here the pages are
//! the driver's instead: when the title reserves its 32-bit graphics
//! memory (one reservation of 185 MB, right after the device is made),
//! `xmem` asks [`title_memory_backed_low`] for a heap of that size whose
//! CPU mapping lies below 4 GiB, and gives the title that mapping's address
//! as its reservation. The title then writes the driver's memory and the
//! card reads it; title buffers and windows in it are buffers placed in
//! that heap ([`place_for`], through `crate::port::place_for`), with no
//! copy, no refresh, no write watch and nothing fetched back, as port 1's
//! are.
//!
//! The heap is, by the switch:
//! - `upload`: `D3D12_HEAP_TYPE_GPU_UPLOAD`, the card's own memory written
//!   by the CPU through the resizable BAR (xcl #2938's mixed segments). The
//!   card reads it at its own speed; the CPU writes it at about 2.5 GiB/s
//!   and READS it at about 25 MiB/s (RTX 4080, driver 610.88).
//! - `custom`: a custom heap in the PC's memory, write-back, pool L0, with
//!   write watching (xcl #2844): the CPU reads and writes it at full speed,
//!   the card over the bus, but as a heap of the driver's.
//!
//! Left alone the driver maps either high, so every free stretch above
//! 4 GiB that could take the mapping is reserved for the moment of the
//! call ("grooming", as xcl's #2844 does), and it can only land below.
//! Which address below does not matter: the title is given the one it got.
//!
//! What it assumes is what port 1 assumes: what our CPU steps read there
//! at submission is what the title's CPU wrote. And: the title maps
//! physical pages there once and keeps them (port 1's notes: never
//! tile-mapped, never remapped); `xmem` leaves later mappings there alone.

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
    /// The buffer over the whole heap whose mapping the title's pages are:
    /// kept, mapped, for good (the mapping lives as long as it does). What
    /// the card copies out of the heap, it copies out of this ([`source`]).
    buffer: ID3D12Resource,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Backed {}

static BACKED: Mutex<Vec<Backed>> = Mutex::new(Vec::new());

/// The lowest start and highest end of every stretch (both 0: none).
static BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Since the start: stretches backed and their bytes; asks where the
/// mapping landed where the title cannot have it; asks the host refused; buffers and windows
/// placed in them; stretches groomed and the reservations that took.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// A heap of the driver's of `len` bytes (a multiple of 64 KiB) whose CPU
/// mapping ends at or below `highest`: the mapping's address, for the
/// caller to give the title as its memory. `None`: nothing was made.
pub fn title_memory_backed_low(len: usize, highest: usize) -> Option<usize> {
    let kind = kind();
    if kind == Kind::Off || len % K64 != 0 || len == 0 {
        return None;
    }
    let host = crate::host::host()?;
    if kind == Kind::Upload && !host.mapped_card_memory() {
        warn_once("VICTORY_UMA_HEAP=upload: this PC offers no GPU_UPLOAD heaps (resizable BAR); title memory stays as it was");
        return None;
    }
    let took = std::time::Instant::now();
    // Left alone the driver maps either kind high in the title's process
    // (the card's memory at the lowest free address in a small one, never
    // here): only below `highest` is anything left that can take it.
    let groomed = groom(0, highest.saturating_add(1), len);
    STATS[6].fetch_add(1, Ordering::Relaxed);
    STATS[7].fetch_add(groomed.len() as u64, Ordering::Relaxed);
    let groom_ms = took.elapsed().as_secs_f64() * 1000.0;
    let made = make(&host.device, kind, len as u64);
    for at in &groomed {
        // SAFETY: our own reservations, nothing in them.
        unsafe { mem::VirtualFree(*at as *mut _, 0, mem::MEM_RELEASE) };
    }
    let (heap, buffer, data) = match made {
        Ok(made) => made,
        Err(e) => {
            STATS[3].fetch_add(1, Ordering::Relaxed);
            tracing::warn!(len, error = %e, ?kind, "UMA heap: the host refused the heap; title memory as before");
            return None;
        }
    };
    let base = data as usize;
    if base == 0 || base % K64 != 0 || base.saturating_add(len - 1) > highest {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            len,
            landed = format_args!("{base:#x}"),
            highest = format_args!("{highest:#x}"),
            groomed = groomed.len(),
            ?kind,
            "UMA heap: the driver mapped the heap where the title cannot have it; title memory as before"
        );
        // SAFETY: our own buffer, mapped once.
        unsafe { buffer.Unmap(0, None) };
        return None;
    }
    crate::host::name(&buffer, || format!("title memory {base:#x}+{len:#x} in a driver heap"));
    let mut backed = BACKED.lock().unwrap_or_else(|e| e.into_inner());
    backed.push(Backed { start: base as u64, end: (base + len) as u64, heap, buffer });
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
        groom_ms,
        ms = took.elapsed().as_secs_f64() * 1000.0,
        "UMA heap: title memory is a driver heap, mapped where the title has it (no copy, no refresh, nothing fetched back)"
    );
    Some(base)
}

/// Whether `[at, at + len)` touches a driver heap of ours (a quick look).
pub(crate) fn touches(at: usize, len: usize) -> bool {
    let hi = BOUNDS[1].load(Ordering::Relaxed);
    hi != 0 && (at as u64) < hi && (at as u64).saturating_add(len as u64) > BOUNDS[0].load(Ordering::Relaxed)
}

/// Whether `[at, at + len)` touches title memory that our code must not
/// read on the CPU: a heap in the card's memory (`upload`), where a read
/// through the BAR runs at about 25 MiB/s. What needs those bytes has the
/// card copy them ([`source`]).
pub(crate) fn card_only(at: usize, len: usize) -> bool {
    touches(at, len) && kind() == Kind::Upload
}

/// For a copy on the card out of title memory `[start, start + len)` that
/// lies wholly in a driver heap of ours: the buffer over that heap and the
/// offset of `start` in it (`CopyBufferRegion`'s source; the buffer is in
/// the common state and stays alive for the run). `None`: not ours.
pub(crate) fn source(start: u64, len: u64) -> Option<(ID3D12Resource, u64)> {
    let hi = BOUNDS[1].load(Ordering::Relaxed);
    if hi == 0 || start >= hi {
        return None;
    }
    let backed = BACKED.lock().unwrap_or_else(|e| e.into_inner());
    let b = backed.iter().find(|b| b.start <= start && start + len.max(1) <= b.end)?;
    Some((b.buffer.clone(), start - b.start))
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

/// Our own CPU reads of title memory that lies in a driver heap of ours,
/// since the last line: by the place in our code that asked
/// (`crate::host::copy_mapped`'s caller), how often and how many bytes.
/// Through the BAR (`upload`) such a read runs at about 25 MiB/s, so with
/// that heap every one of them has to go; this says which are left.
static READS: Mutex<Vec<(&'static str, u32, u64, u64)>> = Mutex::new(Vec::new());

/// All of them since the last line: how many, their bytes.
static READ_ALL: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A read of ours of `len` bytes at `src`, asked for at `from`: counted if
/// it lies in a driver heap of ours.
pub(crate) fn read_on_cpu(src: usize, len: usize, from: &'static std::panic::Location<'static>) {
    if !touches(src, len) {
        return;
    }
    READ_ALL[0].fetch_add(1, Ordering::Relaxed);
    READ_ALL[1].fetch_add(len as u64, Ordering::Relaxed);
    let file = from.file().rsplit(['/', '\\']).next().unwrap_or("?");
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    match reads.iter_mut().find(|r| r.1 == from.line() && r.0 == file) {
        Some(r) => {
            r.2 += 1;
            r.3 += len as u64;
        }
        None => reads.push((file, from.line(), 1, len as u64)),
    }
}

/// [`READS`], taken: the places by bytes, the largest first.
fn reads_taken() -> (u64, u64, Vec<(&'static str, u32, u64, u64)>) {
    let mut reads = std::mem::take(&mut *READS.lock().unwrap_or_else(|e| e.into_inner()));
    reads.sort_by_key(|r| std::cmp::Reverse(r.3));
    (READ_ALL[0].swap(0, Ordering::Relaxed), READ_ALL[1].swap(0, Ordering::Relaxed), reads)
}

/// For the frame-rate line; `frames` since the last one.
pub(crate) fn log(frames: f64) {
    if kind() == Kind::Off {
        return;
    }
    let (n, bytes, reads) = reads_taken();
    let places: Vec<String> = reads
        .iter()
        .take(10)
        .map(|(file, line, n, bytes)| format!("{file}:{line} {:.0}x {:.1} KB", *n as f64 / frames, *bytes as f64 / frames / 1024.0))
        .collect();
    tracing::info!(
        "UMA heap, our own CPU reads of it per frame: {:.0} reads, {:.1} KB; by place: {}",
        n as f64 / frames,
        bytes as f64 / frames / 1024.0,
        if places.is_empty() { "none".to_owned() } else { places.join(", ") }
    );
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

/// Reserves every free stretch of the address space outside `[lo, hi)`
/// that a mapping of `len` bytes could take, keeping a little at the start
/// of each for the driver's own small mappings, and whatever free space
/// touches `[lo, hi)` from outside (so that a mapping cannot lie across its
/// edge): the reservations' addresses, for the caller to release.
fn groom(lo: usize, hi: usize, len: usize) -> Vec<usize> {
    // Less than the mapping, so that it cannot land there.
    let headroom = (len / 4).min(16 << 20).next_multiple_of(K64).min(len.saturating_sub(K64));
    let mut out = Vec::new();
    let mut reserve = |a: usize, b: usize| {
        if b > a {
            // SAFETY: reserves free address space, no pages.
            let at = unsafe { mem::VirtualAlloc(a as *const _, b - a, mem::MEM_RESERVE, mem::PAGE_NOACCESS) };
            if !at.is_null() {
                out.push(at as usize);
            }
        }
    };
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
        let from = info.BaseAddress as usize;
        let to = from.saturating_add(info.RegionSize);
        if info.State == mem::MEM_FREE {
            // Free stretches start on 64 KiB lines (the allocation grain).
            let (start, end) = (from.next_multiple_of(K64), to & !(K64 - 1));
            // The parts of it below and above what is kept free.
            for (a, b) in [(start, end.min(lo)), (start.max(hi), end)] {
                if b <= a {
                    continue;
                }
                let touching = (b == lo && end > lo) || (a == hi && start < hi);
                if touching {
                    reserve(a, b);
                } else if b - a >= len {
                    reserve(a + headroom, b);
                }
            }
        }
        if to <= here {
            break;
        }
        here = to;
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
        let groomed = groom(base, base + len, len);
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

    /// What this PC's card and driver make of it (not WARP): run alone,
    /// under the GPU lock, with `VICTORY_UMA_PROBE_CARD=1`:
    /// `cargo test -p victory-gpu --lib uma::tests::card -- --include-ignored --test-threads=1 --nocapture`.
    #[test]
    #[ignore]
    fn card_probe() {
        if std::env::var_os("VICTORY_UMA_PROBE_CARD").is_none() {
            return;
        }
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: plain device creation on the default adapter.
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).expect("a device") };
        let device = device.unwrap();
        let state = |at: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { mem::VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            format!("state {:#x} type {:#x} protect {:#x} base {:#x} size {:#x}", info.State, info.Type, info.Protect, info.AllocationBase as usize, info.RegionSize)
        };
        for kind in [Kind::Custom, Kind::Upload] {
            // Where the driver puts a mapping left to itself.
            let t = std::time::Instant::now();
            match make(&device, kind, 47 << 20) {
                Ok((_h, b, data)) => {
                    eprintln!("{kind:?} 47 MB, no grooming: mapped at {:#x} in {:.1} ms; {}", data as usize, t.elapsed().as_secs_f64() * 1e3, state(data as usize));
                    // SAFETY: our own buffer.
                    unsafe { b.Unmap(0, None) };
                    drop(b);
                    drop(_h);
                    eprintln!("  after unmap and release: {}", state(data as usize));
                }
                Err(e) => eprintln!("{kind:?} 47 MB, no grooming: {e}"),
            }
            for mb in [47usize, 100, 185] {
                let len = mb << 20;
                // A hole below 4 GiB, as the title's 32-bit memory is.
                let mut req = mem::MEM_ADDRESS_REQUIREMENTS { LowestStartingAddress: (256usize << 20) as *mut _, HighestEndingAddress: (u32::MAX as usize) as *mut _, Alignment: 0 };
                let mut param: mem::MEM_EXTENDED_PARAMETER = unsafe { std::mem::zeroed() };
                param.Anonymous1._bitfield = mem::MemExtendedParameterAddressRequirements as u64;
                param.Anonymous2.Pointer = (&raw mut req).cast();
                // SAFETY: a reservation of free address space, released at once.
                let base = unsafe { mem::VirtualAlloc2(std::ptr::null_mut(), std::ptr::null(), len, mem::MEM_RESERVE, mem::PAGE_NOACCESS, &mut param, 1) } as usize;
                assert_ne!(base, 0, "a hole below 4 GiB");
                // SAFETY: our own reservation.
                unsafe { mem::VirtualFree(base as *mut _, 0, mem::MEM_RELEASE) };
                let t0 = std::time::Instant::now();
                let groomed = groom(base, base + len, len);
                let t1 = std::time::Instant::now();
                let made = make(&device, kind, len as u64);
                let t2 = std::time::Instant::now();
                for at in &groomed {
                    // SAFETY: our own reservations.
                    unsafe { mem::VirtualFree(*at as *mut _, 0, mem::MEM_RELEASE) };
                }
                let t3 = std::time::Instant::now();
                let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
                match made {
                    Ok((heap, buffer, data)) => {
                        eprintln!(
                            "{kind:?} {mb} MB: hole {base:#x}, mapped at {:#x} ({}); groom {:.1} ms ({} reservations), heap+buffer+map {:.1} ms, release {:.1} ms",
                            data as usize,
                            if data as usize == base { "IN THE HOLE" } else { "elsewhere" },
                            ms(t0, t1),
                            groomed.len(),
                            ms(t1, t2),
                            ms(t2, t3)
                        );
                        let n = 16 << 20;
                        let mut scratch = vec![0x5au8; n];
                        let t = std::time::Instant::now();
                        // SAFETY: the mapping is at least `n` bytes.
                        unsafe { std::ptr::copy_nonoverlapping(scratch.as_ptr(), data, n) };
                        let w = t.elapsed().as_secs_f64();
                        let t = std::time::Instant::now();
                        // SAFETY: as above.
                        unsafe { std::ptr::copy_nonoverlapping(data, scratch.as_mut_ptr(), 1 << 20) };
                        let r = t.elapsed().as_secs_f64();
                        eprintln!("  CPU writes {:.0} MiB/s, reads {:.0} MiB/s; {}", 16.0 / w, 1.0 / r, state(data as usize));
                        // Whether the host's write watch can be asked about the mapping.
                        let mut pages = vec![std::ptr::null_mut::<core::ffi::c_void>(); 64];
                        let (mut count, mut grain) = (pages.len(), 0u32);
                        // SAFETY: plain query with room for `count` addresses.
                        let asked = unsafe { mem::GetWriteWatch(1 /* WRITE_WATCH_FLAG_RESET */, data.cast(), len, pages.as_mut_ptr(), &mut count, &mut grain) };
                        // SAFETY: one byte of the mapping.
                        unsafe { data.add(5 * 4096 + 7).write(1) };
                        let mut again = pages.len();
                        // SAFETY: as above.
                        let asked2 = unsafe { mem::GetWriteWatch(0, data.cast(), len, pages.as_mut_ptr(), &mut again, &mut grain) };
                        eprintln!(
                            "  write watch: first asking {} ({} pages written, reset), after one store {} ({} pages, the first at +{:#x})",
                            if asked == 0 { "ok" } else { "REFUSED" },
                            count,
                            if asked2 == 0 { "ok" } else { "REFUSED" },
                            again,
                            if again > 0 && asked2 == 0 { pages[0] as usize - data as usize } else { 0 }
                        );
                        // SAFETY: our own buffer.
                        unsafe { buffer.Unmap(0, None) };
                        drop(buffer);
                        drop(heap);
                        eprintln!("  after unmap and release: {}", state(data as usize));
                    }
                    Err(e) => eprintln!("{kind:?} {mb} MB: hole {base:#x}: REFUSED {e}; groom {:.1} ms ({} reservations), make {:.1} ms; the hole now: {}", ms(t0, t1), groomed.len(), ms(t1, t2), state(base)),
                }
            }
        }
    }
}
