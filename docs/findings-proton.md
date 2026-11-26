# What was learned under Proton on an RX 9070 XT (2026-11-26)

One PC: Ryzen 5 5600X (6 cores, 12 threads), 32 GB, Radeon RX 9070 XT 16 GB, CachyOS,
Proton-CachyOS with vkd3d-proton on RADV, Series S profile, `scripts\play.ps1`
settings. Figures are of diagnostics builds, in city scenes of 4,100-4,500 draws.
Switches were compared with `VICTORY_AB` in one run, phase against neighbouring
phase. Read differences under about 3 % as nothing.

## Where a frame goes

- Median frame in those scenes: 94 ms (about 10.6 fps), and 85 ms (about 11.8 fps)
  with the first three changes measured below.
- **The card is busy 13-14 ms a frame.** The rest of the time it waits for the thread
  that prepares submissions, as on Windows.
- That thread and its helpers spend most of their time in Windows' memory calls, and
  under Wine those are slow (below).

## Why Wine is the wall

Wine runs `VirtualProtect`, `VirtualQuery`, `GetWriteWatch` and its own handling of a
fault in user space, under one lock for the whole process. Every thread that guards,
asks about or faults on title memory waits for it in turn. `VirtualQuery` also walks
the pages of a region one by one, with the lock held. A fault on a guarded block goes
from a Linux signal through Wine's handler, which takes the same lock, to the
exception dispatch and our handler.

Per call, measured with the timeline (`VICTORY_TIMELINE`) in those scenes:

| Call | Under Wine | On Windows |
|---|---|---|
| `VirtualProtect` | 0.2-0.7 ms | about 5 us (`titlewrite.rs`) |
| `VirtualQuery` | 0.3-0.5 ms, up to 4.6 ms | not measured |
| `GetWriteWatch` | 0.4-0.8 ms | not measured |

Little of it is Linux's own work: its `mprotect`, with the flushes of other
processors' address caches, was about 1 % of the samples in `perf`. Most of the time
is waiting for the lock.

## Measured

| What | Change |
|---|---|
| Holes' answers kept by hole, forgotten only where memory changes (`VICTORY_HOLES_ONCE`, 30 frames): questions about holes 6,800 a frame to 36, 175 ms of threads' time to 2.4 | **frame -6.6 %** (faster in 5 of 6 pairs) |
| Comparing pieces with `safecopy::differ` (Wine's `memcmp` is a scalar loop); no new reference for a slot that holds its resource | median -0.7 %, p90 -3.3 % |
| `VICTORY_STORE_BESIDE=1` | -1.7 % (faster in 3 of 4), small |
| `VICTORY_PORT_GUARD=1` against 0 | about 2 % better on, phase medians only |

## Measured and not worth it

- `VICTORY_GUARD_REST` at 20 ms instead of 3: fewer protection changes, but a block
  that rests counts as written and its bytes are compared at every refresh. 1.5 %
  slower (in 5 of 6 pairs).
- `VICTORY_GUARD_HOT=40`: no clear change.
- `VICTORY_COPY_BESIDE=1`: no change.
- Six threads beside the preparing one instead of three: no change. The work waits
  for the lock, not for threads.
- Two frames of submissions in the line (`VICTORY_SUBMIT_FRAMES=2`): slower.
- A write fault on a block another thread has just opened tries the write again before
  asking Windows whether it can be written: half the questions, no change in frames.
  The title's threads are not what a frame waits for.

## Faults found

- Renderer port 1 (`VICTORY_PORT_CONSTANTS=1`) gave black frames: under vkd3d-proton
  a heap opened on a view of a section shows the card other memory. Private memory
  in place works.
- The profiler (`VICTORY_PROFILE`) took the process down at its first sample: the
  unwinder read through a frame register that held no stack address yet.
- Indoor haze with ray tracing off: `VICTORY_AMBIENT_AIR=1.5` clears it.

## Finding where a frame waits

`VICTORY_TIMELINE=<s>[+<s>..],<frames>,<file>.json` records frames from that many
seconds after the start (heavy scenes came after about 180-210 s here). Beside the
Chrome trace it writes a `.txt` summary: what the card's idle time waited on, and
every span and kernel call by count and time. A kernel call has no stack of its own;
the innermost span of ours around it, on the same thread, says who made it.

## What is next

What is left at the top is the protection changes of write faults: about 280 a frame
on the title's threads, about 0.45 ms each. Two ways on:

1. Guard so that far fewer protections change under Wine: a different way of noticing
   the title's writes, chosen when running under Wine (`on_wine` in `xmem.rs` is the
   precedent).
2. A Wine whose memory calls do not all wait for one lock. That would help every game,
   but it is Wine's to change.
