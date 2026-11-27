//! Pixel quad reads drawn on the host: helper lanes at a shared triangle edge, quad-local
//! DPP8, sources outside EXEC and whole-quad mode. From fxceless/victory-fork#1408
//! (ByFxbian, after DexNom/VIctory 921d8d3b).

use super::barycentric_tests::{element, pixel_with_inputs};
use super::bitcode::{Attr, Term};
use super::host::{Draw, Host};
use super::shader::Shader;
use windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_PIXEL;

/// Pass clip-space coordinates through to SV_Position.
fn vertex() -> Vec<u8> {
    let mut vs = Shader::new(false);
    let mut input = element("POSITION", 0, 0);
    input.usage = 15;
    vs.inputs.push(input);
    let mut output = element("SV_Position", 3, 0);
    output.usage = 15;
    output.interp = 4;
    vs.outputs.push(output);
    let zero = vs.m.i32(0);
    let undef = vs.m.undef(vs.t.i32);
    for col in 0..4 {
        let col = vs.m.i8(col);
        let value = vs.op(
            vs.entry,
            "loadInput.f32",
            Attr::ReadNone,
            vs.t.f32,
            4,
            &[zero, zero, col, undef],
        );
        vs.op(
            vs.entry,
            "storeOutput.f32",
            Attr::NoUnwind,
            vs.t.void,
            5,
            &[zero, zero, col, value],
        );
    }
    vs.m.terminate(vs.entry, Term::Ret);
    vs.finish()
}

const WQM: u32 = 0xBEFE_097E;
const RESTORE_EXEC: u32 = 0xBEFE_0300;

fn dpp(dst: u32, src: u32, pattern: u32, flags: u32) -> [u32; 2] {
    [0x7E00_02FA | dst << 17, flags | pattern << 8 | src]
}

fn dpp8(dst: u32, src: u32, lanes: [u32; 4], fi: bool) -> [u32; 2] {
    let sel = (0..8).fold(0, |out, i| out | (lanes[i % 4] | (i as u32 & 4)) << (3 * i));
    [
        0x7E00_0200 | dst << 17 | (0xE9 + u32::from(fi)),
        sel << 8 | src,
    ]
}

fn swizzle(dst: u32, src: u32, offset: u32) -> [u32; 2] {
    [0xD8D4_0000 | offset, dst << 24 | src]
}

fn mov(dst: u32, src: u32) -> u32 {
    0x7E00_0200 | dst << 17 | src
}

fn narrow_exec(words: &mut Vec<u32>) {
    words.extend([0xBE80_037E, 0x7C28_00FF, 7.0f32.to_bits()]); // save EXEC; v_cmpx_gt_f32 7, v0
}

fn render(
    label: &str,
    words: &[u32],
    expected: impl Fn(usize, usize) -> [f32; 4],
) -> Result<(), String> {
    use windows::Win32::Graphics::Dxgi::*;
    let mut words = words.to_vec();
    words.extend([0xF800_180F, 0x0504_0302, 0xBF81_0000]); // export v2..v5
    let mut ps = pixel_with_inputs(&words, (1 << 8) | (1 << 9), &[element("SV_Position", 3, 0)])
        .map_err(|e| format!("{label}: translation failed: {e}"))?;
    super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
        .expect("dxil.dll required")
        .validate(&mut ps.dxil)
        .expect("DXIL validation");
    eprintln!("{label}: {} bytes of validated DXIL", ps.dxil.len());
    let mut ran = 0;
    let vs = vertex();

    const SIZE: usize = 16;
    let a = [-1.0, -1.0, 0.5, 1.0];
    let b = [-1.0, 1.0, 0.5, 1.0];
    let c = [1.0, -1.0, 0.5, 1.0];
    let d = [1.0, 1.0, 0.5, 1.0];
    let mut failures = Vec::new();
    for (name, warp) in [("WARP", true), ("hardware", false)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {name} device; skipped");
            continue;
        };
        // SAFETY: queries the live device's adapter by LUID.
        let desc = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            factory
                .EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid())
                .unwrap()
                .GetDesc1()
                .unwrap()
        };
        assert_eq!(desc.Flags & 2 != 0, warp);
        let adapter = String::from_utf16_lossy(&desc.Description);
        eprintln!(
            "{label} on {name}: {}",
            adapter.trim_end_matches(char::from(0))
        );
        ran += 1;
        let root = host
            .root_signature_for(&[(&ps, D3D12_SHADER_VISIBILITY_PIXEL)])
            .unwrap();
        let pso = host.pipeline_with(&root, &vs, &ps.dxil).unwrap();
        // The harness draws one triangle per call into a cleared target. Add
        // the targets so every pixel must be owned by exactly one triangle.
        let halves = [[a, b, c], [b, c, d]].map(|triangle| {
            host.draw(
                &root,
                &pso,
                &Draw {
                    roots: Vec::new(),
                    textures: &[],
                    samplers: &[],
                    vertices: triangle.as_flattened(),
                    stride: 16,
                    size: (SIZE as u32, SIZE as u32),
                    mesh_groups: 0,
                },
            )
            .unwrap()
        });
        let mut mismatches = Vec::new();
        for y in 0..SIZE {
            for x in 0..SIZE {
                let i = y * SIZE + x;
                let pixel: [f32; 4] = std::array::from_fn(|c| halves[0][i][c] + halves[1][i][c]);
                let expected = expected(x, y);
                if pixel != expected {
                    mismatches.push((x, y, pixel, expected));
                }
            }
        }
        if let Some(first) = mismatches.first() {
            let edge = mismatches
                .iter()
                .filter(|(x, y, _, _)| x / 2 == y / 2)
                .count();
            failures.push(format!("{name} {label}: {} incorrect pixels, {edge} in quads spanning the diagonal; first (x, y, RGBA, expected) {first:?}", mismatches.len()));
        }
    }
    assert!(ran > 0, "no SM 6.6 draw ran");
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

#[test]
fn pixel_quad_broadcast_reads_helper_coordinates_at_shared_triangle_edge() {
    let mut results = Vec::new();
    for is_dpp in [false, true] {
        let label = if is_dpp {
            "dpp-broadcast"
        } else {
            "ds-broadcast"
        };
        let mut words = vec![WQM];
        for (lane, src) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
            let pat = lane * 0x55;
            words.extend(if is_dpp {
                dpp(lane + 2, src, pat, 0xFF00_0000)
            } else {
                swizzle(lane + 2, src, 0x8000 | pat)
            });
        }
        results.push(render(label, &words, |x, y| {
            let left = (x & !1) as f32 + 0.5;
            let bottom = (y & !1) as f32 + 1.5;
            [left, left + 1.0, bottom, bottom]
        }));
    }
    check(results);
}

fn check(results: Vec<Result<(), String>>) {
    let failures: Vec<_> = results.into_iter().filter_map(Result::err).collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn pixel_quad_exchanges_and_dpp8_keep_their_quad() {
    let mut results = Vec::new();
    for (kind, label) in ["ds-exchanges", "dpp-exchanges", "dpp8-exchanges"]
        .into_iter()
        .enumerate()
    {
        let mut words = vec![WQM];
        for (dst, src, xor) in [(2, 0, 1), (3, 1, 2), (4, 0, 3), (5, 0, 0)] {
            let lanes = [0, 1, 2, 3].map(|i| i ^ xor);
            let pattern = lanes.iter().enumerate().map(|(i, n)| n << (2 * i)).sum();
            words.extend(match kind {
                0 => swizzle(dst, src, 0x1f | xor << 10),
                1 => dpp(dst, src, pattern, 0xFF00_0000),
                _ => dpp8(dst, src, lanes, false),
            });
        }
        results.push(render(label, &words, |x, y| {
            [
                (x ^ 1) as f32 + 0.5,
                (y ^ 1) as f32 + 0.5,
                (x ^ 1) as f32 + 0.5,
                x as f32 + 0.5,
            ]
        }));
    }
    check(results);
}

#[test]
fn pixel_quad_mixed_permutations_read_each_selected_lane() {
    let lanes = [0, 3, 1, 2];
    let pattern = lanes
        .iter()
        .enumerate()
        .map(|(i, n)| n << (2 * i))
        .sum::<u32>();
    let mut words = vec![WQM];
    words.extend(dpp(2, 0, pattern, 0xFF00_0000));
    words.extend(dpp8(3, 1, lanes, false));
    words.extend(swizzle(4, 0, 0x8000 | pattern));
    words.push(mov(5, 257));
    render("mixed-permutations", &words, |x, y| {
        let selected = lanes[(y & 1) * 2 + (x & 1)] as usize;
        let sx = ((x & !1) | (selected & 1)) as f32 + 0.5;
        let sy = ((y & !1) | (selected >> 1)) as f32 + 0.5;
        [sx, sy, sx, y as f32 + 0.5]
    })
    .unwrap();
}

#[test]
fn pixel_dpp_respects_exec_fetch_inactive_and_bound_control() {
    let mut words = vec![WQM];
    words.extend((2..6).map(|dst| mov(dst, 242))); // sentinel 1.0
    narrow_exec(&mut words);
    words.extend(dpp(2, 0, 0xb1, 0xFF00_0000)); // bound off
    words.extend(dpp(3, 0, 0xb1, 0xFF08_0000)); // bound on
    words.extend(dpp(4, 0, 0xb1, 0xFF04_0000)); // fetch inactive
    words.extend(dpp(5, 0, 0xb1, 0xF000_0000)); // no banks
    words.push(RESTORE_EXEC);
    render("dpp-exec", &words, |x, _| match x {
        0..6 => {
            let partner = (x ^ 1) as f32 + 0.5;
            [partner, partner, partner, 1.0]
        }
        6 => [1.0, 0.0, 7.5, 1.0],
        _ => [1.0; 4],
    })
    .unwrap();
}

#[test]
fn pixel_wqm_expands_exec_only_in_quads_with_live_lanes() {
    let mut words: Vec<_> = (2..6).map(|dst| mov(dst, 128)).collect();
    narrow_exec(&mut words);
    words.extend([WQM, mov(2, 242), RESTORE_EXEC]);
    render("wqm-exec", &words, |x, _| {
        [if x < 8 { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0]
    })
    .unwrap();
}

#[test]
fn pixel_dpp8_and_ds_swizzle_agree_on_inactive_sources() {
    let mut words = vec![WQM];
    words.extend((2..6).map(|dst| mov(dst, 242)));
    narrow_exec(&mut words);
    words.extend(dpp8(2, 0, [1, 0, 3, 2], false));
    words.extend(dpp8(3, 0, [1, 0, 3, 2], true));
    words.extend(swizzle(4, 0, 0x8000 | 0xb1));
    words.extend(dpp(5, 0, 0xb1, 0x0F00_0000)); // no rows
    words.push(RESTORE_EXEC);
    render("dpp8-ds-exec", &words, |x, _| match x {
        0..6 => {
            let partner = (x ^ 1) as f32 + 0.5;
            [partner, partner, partner, 1.0]
        }
        6 => [0.0, 7.5, 0.0, 1.0],
        _ => [1.0; 4],
    })
    .unwrap();
}

#[test]
fn pixel_dpp8_rejects_a_cross_quad_source() {
    let mut words = vec![WQM];
    words.extend(dpp8(2, 0, [4, 0, 3, 2], false));
    words.extend([
        mov(3, 256),
        mov(4, 256),
        mov(5, 256),
        0xF800_180F,
        0x0504_0302,
        0xBF81_0000,
    ]);
    let rejected = pixel_with_inputs(&words, (1 << 8) | (1 << 9), &[element("SV_Position", 3, 0)]);
    assert!(matches!(
        rejected,
        Err(crate::ShaderError::Unsupported {
            what: "a cross-lane operation outside compute",
            ..
        })
    ));
}
