//! Copies between a texture in a 16-bit colour format and a target whose
//! host copy has 8-bit RGBA standing in for that format
//! ([`crate::resource::target_format`]). The host cannot copy between the
//! two: their texels differ in size. Such a copy goes through two buffers
//! of ours instead: the source's texels are copied into the first, a
//! compute pass of ours widens (16 to 32 bits) or narrows (32 to 16 bits)
//! each texel into the second, and the second is copied to the
//! destination.
//!
//! The conversion is the one the host makes between UNORM formats: each
//! channel goes through its value in [0, 1] and is rounded to the nearest
//! level. The tests hold the pass to that conversion on the CPU
//! (`widen` and `narrow`); [`SOURCE`] is the same on the GPU.

/// The console's (and the host's) B5G5R5A1, `DXGI_FORMAT_B5G5R5A1_UNORM`.
#[cfg(test)]
pub(crate) const B5G5R5A1: u32 = 86;
/// The console's (and the host's) B4G4R4A4, `DXGI_FORMAT_B4G4R4A4_UNORM`.
pub(crate) const B4G4R4A4: u32 = 115;

/// The bits of each channel of a 16-bit colour format, from the lowest
/// bit up: blue, green, red, alpha.
#[cfg(test)]
const fn bits(format: u32) -> Option<[u32; 4]> {
    match format {
        B5G5R5A1 => Some([5, 5, 5, 1]),
        B4G4R4A4 => Some([4, 4, 4, 4]),
        _ => None,
    }
}

/// Where each channel of [`bits`] goes in an 8-bit RGBA texel (red in the
/// lowest byte): blue, green, red, alpha.
#[cfg(test)]
const SHIFTS: [u32; 4] = [16, 8, 0, 24];

/// A texel of `format` (B5G5R5A1 or B4G4R4A4) as 8-bit RGBA, red in the
/// lowest byte; `None` for another format.
#[cfg(test)]
pub(crate) fn widen(format: u32, texel: u16) -> Option<u32> {
    let bits = bits(format)?;
    let (mut out, mut at) = (0u32, 0u32);
    for (n, shift) in bits.into_iter().zip(SHIFTS) {
        let max = (1 << n) - 1;
        let level = (u32::from(texel) >> at) & max;
        out |= ((level * 255 + max / 2) / max) << shift;
        at += n;
    }
    Some(out)
}

/// 8-bit RGBA (red in the lowest byte) as a texel of `format` (B5G5R5A1 or
/// B4G4R4A4); `None` for another format.
#[cfg(test)]
pub(crate) fn narrow(format: u32, rgba: u32) -> Option<u16> {
    let bits = bits(format)?;
    let (mut out, mut at) = (0u32, 0u32);
    for (n, shift) in bits.into_iter().zip(SHIFTS) {
        let max = (1 << n) - 1;
        let level = (rgba >> shift) & 255;
        out |= ((level * max + 127) / 255) << at;
        at += n;
    }
    Some(out as u16)
}

/// The compute pass: one thread for each texel it widens, or for each pair
/// of texels it narrows (one 32-bit word of output, so no two threads
/// write the same word). Rows are `pitch` bytes apart in both buffers;
/// a 3D region's slices follow one another, `rows` in all.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) const SOURCE: &str = r#"
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);
cbuffer Args : register(b0) {
    uint width;
    uint rows;
    uint src_pitch;
    uint dst_pitch;
    uint narrowing;
    uint four_bits;
};

// Blue, green, red, alpha from the lowest bit up.
uint4 channel_bits() {
    return four_bits != 0 ? uint4(4, 4, 4, 4) : uint4(5, 5, 5, 1);
}

// Where each of those goes in 8-bit RGBA, red in the lowest byte.
static const uint4 SHIFTS = uint4(16, 8, 0, 24);

uint widen(uint texel) {
    uint4 bits = channel_bits();
    uint out_ = 0;
    uint at = 0;
    [unroll] for (uint i = 0; i < 4; i++) {
        uint m = (1u << bits[i]) - 1;
        uint level = (texel >> at) & m;
        out_ |= ((level * 255 + m / 2) / m) << SHIFTS[i];
        at += bits[i];
    }
    return out_;
}

uint narrow(uint rgba) {
    uint4 bits = channel_bits();
    uint out_ = 0;
    uint at = 0;
    [unroll] for (uint i = 0; i < 4; i++) {
        uint m = (1u << bits[i]) - 1;
        uint level = (rgba >> SHIFTS[i]) & 255;
        out_ |= ((level * m + 127) / 255) << at;
        at += bits[i];
    }
    return out_;
}

[RootSignature("RootConstants(num32BitConstants=6, b0), UAV(u0), UAV(u1)")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.y >= rows) {
        return;
    }
    if (narrowing != 0) {
        uint x = id.x * 2;
        if (x >= width) {
            return;
        }
        uint row = id.y * src_pitch + x * 4;
        uint low = narrow(src.Load(row));
        uint high = x + 1 < width ? narrow(src.Load(row + 4)) : 0;
        dst.Store(id.y * dst_pitch + id.x * 4, low | (high << 16));
    } else {
        uint x = id.x;
        if (x >= width) {
            return;
        }
        uint word = src.Load(id.y * src_pitch + (x & ~1u) * 2);
        dst.Store(id.y * dst_pitch + x * 4, widen((word >> ((x & 1) * 16)) & 0xffff));
    }
}
"#;

/// The size of a copy region in texels.
#[cfg_attr(not(windows), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Extent {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth: u32,
}

/// The bytes between rows of `width` texels of `bytes` each in a buffer
/// the host copies textures to or from (256-byte aligned).
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn pitch(width: u32, bytes: u32) -> Option<u32> {
    width.checked_mul(bytes)?.checked_next_multiple_of(256)
}

/// The thread groups of the pass for a region (8 by 8 threads each).
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn groups(extent: Extent, narrowing: bool) -> [u32; 2] {
    let threads = if narrowing {
        extent.width.div_ceil(2)
    } else {
        extent.width
    };
    [
        threads.div_ceil(8),
        (extent.height * extent.depth).div_ceil(8),
    ]
}

/// The extent of subresource `sub` of a texture (`width`, `height`,
/// `depth_or_array` and `mips` as in its description; `volume` for a 3D
/// texture): the mip's size, with one slice unless it is 3D.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn subresource_extent(
    [width, height, depth_or_array, mips]: [u64; 4],
    volume: bool,
    sub: u32,
) -> Extent {
    let mip = sub % (mips.max(1) as u32);
    let at = |size: u64| ((size >> mip).max(1)).min(u64::from(u32::MAX)) as u32;
    Extent {
        width: at(width),
        height: at(height),
        depth: if volume { at(depth_or_array) } else { 1 },
    }
}

#[cfg(windows)]
pub(crate) use host::{ready, record, subresources};

#[cfg(windows)]
mod host {
    use super::{Extent, SOURCE, groups, pitch};
    use crate::host::{HostList, transition};
    use std::sync::OnceLock;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

    /// The root signature and pipeline of the pass, made at the first use;
    /// `None` if they could not be made (logged once).
    fn pass() -> Option<&'static (ID3D12RootSignature, ID3D12PipelineState)> {
        static PASS: OnceLock<Option<(ID3D12RootSignature, ID3D12PipelineState)>> = OnceLock::new();
        PASS.get_or_init(|| {
            let host = crate::host::host()?;
            let made = (|| -> windows::core::Result<_> {
                let code = crate::host::compile(SOURCE, windows::core::s!("main"), windows::core::s!("cs_5_1"))?;
                // SAFETY: our own compiled bytes, the root signature
                // inside them.
                unsafe {
                    let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code)?;
                    let pso: ID3D12PipelineState =
                        host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                            pRootSignature: std::mem::transmute_copy(&root),
                            CS: D3D12_SHADER_BYTECODE {
                                pShaderBytecode: code.as_ptr().cast(),
                                BytecodeLength: code.len(),
                            },
                            ..Default::default()
                        })?;
                    Ok((root, pso))
                }
            })();
            match made {
                Ok(pass) => Some(pass),
                Err(error) => {
                    tracing::warn!(%error, "no conversion for copies with 8-bit RGBA standing in: the pass could not be made");
                    None
                }
            }
        })
        .as_ref()
    }

    /// Whether the pass can be made (made now if not yet).
    pub(crate) fn ready() -> bool {
        pass().is_some()
    }

    /// How many subresources a copy of the whole of `texture` has (planes
    /// left out: the formats converted have one).
    pub(crate) fn subresources(texture: &ID3D12Resource) -> u32 {
        // SAFETY: a plain query.
        let d = unsafe { texture.GetDesc() };
        let mips = u32::from(d.MipLevels.max(1));
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            mips
        } else {
            mips * u32::from(d.DepthOrArraySize.max(1))
        }
    }

    /// The extent of subresource `sub` of `texture`.
    fn extent(texture: &ID3D12Resource, sub: u32) -> Extent {
        // SAFETY: a plain query.
        let d = unsafe { texture.GetDesc() };
        super::subresource_extent(
            [
                d.Width,
                u64::from(d.Height),
                u64::from(d.DepthOrArraySize),
                u64::from(d.MipLevels),
            ],
            d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D,
            sub,
        )
    }

    pub(super) fn subresource(texture: &ID3D12Resource, index: u32) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: index,
            },
        }
    }

    pub(super) fn rows(
        buffer: &ID3D12Resource,
        format: DXGI_FORMAT,
        extent: Extent,
        pitch: u32,
    ) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: extent.width,
                        Height: extent.height,
                        Depth: extent.depth,
                        RowPitch: pitch,
                    },
                },
            },
        }
    }

    /// Records into `list` a copy from subresource `from_sub` of `from`
    /// (the box `area` of it, or all of it) to subresource `to_sub` of `to`
    /// at `at`, converting the texels between 16-bit `format` (B5G5R5A1 or
    /// B4G4R4A4) and 8-bit RGBA: narrowing them if `from` is the 8-bit
    /// one, widening them if `to` is. Both textures are already in their
    /// copy states. The pass binds its own root signature and pipeline
    /// (the caller tells the list's draw state). `false` if nothing was
    /// recorded (no pass, no buffers, a multisampled texture, or an empty
    /// region).
    #[allow(clippy::too_many_arguments)] // two locations, a box and the format
    pub(crate) fn record(
        list: &HostList,
        format: u32,
        to: &ID3D12Resource,
        to_sub: u32,
        at: [u32; 3],
        from: &ID3D12Resource,
        from_sub: u32,
        area: Option<D3D12_BOX>,
    ) -> bool {
        let (Some(host), Some((root, pso))) = (crate::host::host(), pass()) else {
            return false;
        };
        // SAFETY: plain queries.
        let (to_desc, from_desc) = unsafe { (to.GetDesc(), from.GetDesc()) };
        if to_desc.SampleDesc.Count > 1 || from_desc.SampleDesc.Count > 1 {
            return false;
        }
        let narrowing = from_desc.Format.0 as u32 != format;
        let extent = match area {
            Some(b) => Extent {
                width: b.right.saturating_sub(b.left),
                height: b.bottom.saturating_sub(b.top),
                depth: b.back.saturating_sub(b.front),
            },
            None => extent(from, from_sub),
        };
        if extent.width == 0 || extent.height == 0 || extent.depth == 0 {
            return false;
        }
        let (src_bytes, dst_bytes) = if narrowing { (4, 2) } else { (2, 4) };
        let (Some(src_pitch), Some(dst_pitch)) = (
            pitch(extent.width, src_bytes),
            pitch(extent.width, dst_bytes),
        ) else {
            return false;
        };
        let rows_in_all = u64::from(extent.height) * u64::from(extent.depth);
        let (Some(a), Some(b)) = (
            host.uav_buffer(u64::from(src_pitch) * rows_in_all),
            host.uav_buffer(u64::from(dst_pitch) * rows_in_all),
        ) else {
            return false;
        };
        list.keep(&a);
        list.keep(&b);
        let constants = [
            extent.width,
            extent.height * extent.depth,
            src_pitch,
            dst_pitch,
            u32::from(narrowing),
            u32::from(format == super::B4G4R4A4),
        ];
        let [x, y] = groups(extent, narrowing);
        let source = subresource(from, from_sub);
        let into_a = rows(&a, from_desc.Format, extent, src_pitch);
        let from_b = rows(&b, to_desc.Format, extent, dst_pitch);
        let destination = subresource(to, to_sub);
        let area = area.map(|b| D3D12_BOX {
            right: b.left + extent.width,
            bottom: b.top + extent.height,
            back: b.front + extent.depth,
            ..b
        });
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            format!(
                "16-bit format {format} copy through 8-bit RGBA, {extent:?}, narrowing {narrowing}"
            )
        });
        // SAFETY: an open list; both textures live in their copy states
        // (the caller's), our buffers kept by the list until it is reset;
        // the root arguments match the pass's root signature.
        unsafe {
            let l = &list.list;
            l.ResourceBarrier(&[transition(
                &a,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            l.CopyTextureRegion(
                &into_a,
                0,
                0,
                0,
                &source,
                area.as_ref().map(std::ptr::from_ref),
            );
            l.ResourceBarrier(&[
                transition(
                    &a,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
                transition(
                    &b,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);
            l.SetComputeRootSignature(root);
            l.SetPipelineState(pso);
            l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
            l.SetComputeRootUnorderedAccessView(1, a.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(2, b.GetGPUVirtualAddress());
            l.Dispatch(x, y, 1);
            l.ResourceBarrier(&[transition(
                &b,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyTextureRegion(&destination, at[0], at[1], at[2], &from_b, None);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host's UNORM conversion: through [0, 1], rounded to nearest.
    fn through_float(level: u32, from_max: u32, to_max: u32) -> u32 {
        (f64::from(level) / f64::from(from_max) * f64::from(to_max)).round() as u32
    }

    #[test]
    fn widening_is_the_unorm_conversion() {
        for format in [B5G5R5A1, B4G4R4A4] {
            let bits = bits(format).unwrap();
            for texel in 0..=u16::MAX {
                let rgba = widen(format, texel).unwrap();
                let mut at = 0;
                for (n, shift) in bits.into_iter().zip(SHIFTS) {
                    let max = (1 << n) - 1;
                    let level = (u32::from(texel) >> at) & max;
                    assert_eq!(
                        (rgba >> shift) & 255,
                        through_float(level, max, 255),
                        "{format} {texel:#06x}"
                    );
                    at += n;
                }
            }
        }
    }

    #[test]
    fn narrowing_is_the_unorm_conversion() {
        for format in [B5G5R5A1, B4G4R4A4] {
            let bits = bits(format).unwrap();
            for level in 0..=255u32 {
                // Every channel at `level`, and each alone.
                let all = level * 0x0101_0101;
                let texel = u32::from(narrow(format, all).unwrap());
                let mut at = 0;
                for n in bits {
                    let max = (1 << n) - 1;
                    assert_eq!(
                        (texel >> at) & max,
                        through_float(level, 255, max),
                        "{format} {level}"
                    );
                    at += n;
                }
                for (i, shift) in SHIFTS.into_iter().enumerate() {
                    let texel = u32::from(narrow(format, level << shift).unwrap());
                    let at: u32 = bits[..i].iter().sum();
                    let max = (1 << bits[i]) - 1;
                    assert_eq!(
                        texel,
                        through_float(level, 255, max) << at,
                        "{format} {level} channel {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn every_texel_survives_widening_and_narrowing() {
        for format in [B5G5R5A1, B4G4R4A4] {
            for texel in 0..=u16::MAX {
                let back = narrow(format, widen(format, texel).unwrap()).unwrap();
                // B5G5R5A1's alpha is the top bit; every bit is a channel's.
                assert_eq!(back, texel, "{format} {texel:#06x}");
            }
        }
    }

    #[test]
    fn known_texels() {
        // Opaque pure red, green and blue, and transparent white.
        assert_eq!(widen(B5G5R5A1, 0xfc00), Some(0xff00_00ff));
        assert_eq!(widen(B5G5R5A1, 0x83e0), Some(0xff00_ff00));
        assert_eq!(widen(B5G5R5A1, 0x801f), Some(0xffff_0000));
        assert_eq!(widen(B5G5R5A1, 0x7fff), Some(0x00ff_ffff));
        assert_eq!(widen(B4G4R4A4, 0xff00), Some(0xff00_00ff));
        assert_eq!(widen(B4G4R4A4, 0x0fff), Some(0x00ff_ffff));
        assert_eq!(widen(B4G4R4A4, 0x1234), Some(0x1144_3322));
        assert_eq!(narrow(B4G4R4A4, 0x1144_3322), Some(0x1234));
        assert_eq!(widen(28, 0), None);
        assert_eq!(narrow(28, 0), None);
    }

    #[test]
    fn rows_are_pitched_for_the_host() {
        assert_eq!(pitch(1, 2), Some(256));
        assert_eq!(pitch(128, 2), Some(256));
        assert_eq!(pitch(129, 2), Some(512));
        assert_eq!(pitch(1280, 4), Some(5120));
        assert_eq!(pitch(u32::MAX, 4), None);
    }

    #[test]
    fn groups_cover_every_texel_once() {
        let e = |width, height, depth| Extent {
            width,
            height,
            depth,
        };
        assert_eq!(groups(e(1, 1, 1), false), [1, 1]);
        assert_eq!(groups(e(1, 1, 1), true), [1, 1]);
        assert_eq!(groups(e(1280, 720, 1), false), [160, 90]);
        assert_eq!(groups(e(1280, 720, 1), true), [80, 90]);
        assert_eq!(groups(e(17, 9, 2), true), [2, 3]);
    }

    #[test]
    fn subresource_extents_follow_mips_and_slices() {
        // 1280x720, six array slices, three mips: subresource 4 is mip 1
        // of slice 1.
        assert_eq!(
            subresource_extent([1280, 720, 6, 3], false, 4),
            Extent {
                width: 640,
                height: 360,
                depth: 1
            }
        );
        // A 3D texture keeps its depth, halved per mip, never below 1.
        assert_eq!(
            subresource_extent([4, 4, 8, 4], true, 3),
            Extent {
                width: 1,
                height: 1,
                depth: 1
            }
        );
        assert_eq!(
            subresource_extent([4, 4, 8, 4], true, 1),
            Extent {
                width: 2,
                height: 2,
                depth: 4
            }
        );
    }

    /// The pass on the host GPU gives what [`widen`] and [`narrow`] give,
    /// for every 16-bit texel and both formats (run where a host or WARP
    /// is available).
    #[cfg(windows)]
    #[test]
    fn the_pass_converts_as_the_cpu_does() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = crate::host::host() else {
            return;
        };
        // 256 by 256 texels: every 16-bit value once.
        let side = 256u32;
        for format in [B5G5R5A1, B4G4R4A4] {
            let make = |fmt: DXGI_FORMAT| {
                let heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: u64::from(side),
                    Height: side,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: fmt,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                };
                let mut texture: Option<ID3D12Resource> = None;
                // SAFETY: plain creation.
                unsafe {
                    host.device
                        .CreateCommittedResource(
                            &heap,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut texture,
                        )
                        .unwrap()
                };
                texture.unwrap()
            };
            let small = make(DXGI_FORMAT(format as i32));
            let wide = make(DXGI_FORMAT_R8G8B8A8_UNORM);
            let narrowed = make(DXGI_FORMAT(format as i32));
            // Fill the 16-bit texture with every value, row by row.
            let pitch16 = pitch(side, 2).unwrap();
            let (upload, data) = host.upload_buffer(u64::from(pitch16 * side)).unwrap();
            for v in 0..=u16::MAX {
                let (x, y) = (u32::from(v) % side, u32::from(v) / side);
                // SAFETY: inside the mapped upload buffer.
                unsafe {
                    data.add((y * pitch16 + x * 2) as usize)
                        .cast::<u16>()
                        .write_unaligned(v)
                };
            }
            let extent = Extent {
                width: side,
                height: side,
                depth: 1,
            };
            let list = host.new_list().unwrap();
            let pitch32 = pitch(side, 4).unwrap();
            let readback = host.read_back_buffer(u64::from(pitch32 * side)).unwrap();
            let readback16 = host.read_back_buffer(u64::from(pitch16 * side)).unwrap();
            // SAFETY: an open list and live resources in the states named.
            unsafe {
                let l = &list.list;
                l.CopyTextureRegion(
                    &host::subresource(&small, 0),
                    0,
                    0,
                    0,
                    &host::rows(&upload, DXGI_FORMAT(format as i32), extent, pitch16),
                    None,
                );
                l.ResourceBarrier(&[crate::host::transition(
                    &small,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            assert!(record(&list, format, &wide, 0, [0; 3], &small, 0, None));
            // SAFETY: as above.
            unsafe {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &wide,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            assert!(record(&list, format, &narrowed, 0, [0; 3], &wide, 0, None));
            // SAFETY: as above.
            unsafe {
                let l = &list.list;
                l.ResourceBarrier(&[crate::host::transition(
                    &narrowed,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                l.CopyTextureRegion(
                    &host::rows(&readback, DXGI_FORMAT_R8G8B8A8_UNORM, extent, pitch32),
                    0,
                    0,
                    0,
                    &host::subresource(&wide, 0),
                    None,
                );
                l.CopyTextureRegion(
                    &host::rows(&readback16, DXGI_FORMAT(format as i32), extent, pitch16),
                    0,
                    0,
                    0,
                    &host::subresource(&narrowed, 0),
                    None,
                );
            }
            list.close();
            host.execute(&[&list]);
            let mapped = |buffer: &ID3D12Resource| {
                let mut bytes = std::ptr::null_mut();
                // SAFETY: a read-back buffer the finished list wrote.
                unsafe { buffer.Map(0, None, Some(&mut bytes)).unwrap() };
                bytes.cast::<u8>().cast_const()
            };
            let (wide_bytes, narrow_bytes) = (mapped(&readback), mapped(&readback16));
            for v in 0..=u16::MAX {
                let (x, y) = (u32::from(v) % side, u32::from(v) / side);
                // SAFETY: inside the mapped read-back buffers.
                let (got, back) = unsafe {
                    (
                        wide_bytes
                            .add((y * pitch32 + x * 4) as usize)
                            .cast::<u32>()
                            .read_unaligned(),
                        narrow_bytes
                            .add((y * pitch16 + x * 2) as usize)
                            .cast::<u16>()
                            .read_unaligned(),
                    )
                };
                assert_eq!(
                    Some(got),
                    widen(format, v),
                    "format {format}, texel {v:#06x} widened"
                );
                assert_eq!(back, v, "format {format}, texel {v:#06x} narrowed back");
            }
        }
    }
}
