# Implementation Notes

Technical documentation for developers working on or extending sba-solver-wasm.

## Architecture

```
sba-solver-wasm/
├── src/lib.rs              # WASM interface (~650 lines)
├── apex-solver-fork/       # Modified apex-solver for WASM
├── pkg/                    # Built WASM module (generated)
├── tests/                  # Playwright browser tests
└── examples/               # Interactive browser demo
```

## apex-solver Fork

This module uses a fork of [apex-solver](https://github.com/amin-abouee/apex-solver), a Rust optimization library for bundle adjustment and SLAM. The fork lives in `apex-solver-fork/` and contains modifications for WASM compatibility.

### Changes from Upstream

**Feature flags added:**
- `io` - File I/O operations (disabled in WASM)
- `parallel` - Rayon parallelization (disabled in WASM)
- `cli` - Command-line tools
- `logging` - Tracing/logging
- `visualization` - Rerun visualization

**Dependency changes:**
- `memmap2` - Made optional, gated behind `io` feature
- `rayon` - Made optional, gated behind `parallel` feature
- `std::time::Instant` - Replaced with `web-time` crate for WASM timing

**Code changes:**
- Conditional compilation for parallel vs sequential iteration
- Feature-gated `IoError` in error module
- Updated to Rust 2024 edition for let-chain syntax

### WASM Build Configuration

`.cargo/config.toml`:
```toml
[target.wasm32-unknown-unknown]
rustflags = [
    "-C", "target-feature=+simd128",
    "--cfg", "getrandom_backend=\"wasm_js\""
]
```

Key settings:
- SIMD128 enabled for performance
- `getrandom` configured with `wasm_js` backend for random number generation

## Custom ReprojectionFactor

The core of this module is a custom `ReprojectionFactor` implementation in `src/lib.rs` that integrates with apex-solver's optimization framework.

### What it does

1. **Transforms 3D points** from world to camera coordinates using SE3 pose
2. **Projects points** using pinhole model with radial-tangential distortion
3. **Computes analytical Jacobians** with respect to pose (6 DOF) and point (3 DOF)

### Reprojection Model

```
residual = project(R * point_world + t) - observed_2d
```

Projection with Brown-Conrady distortion:
```
x' = x/z, y' = y/z                         # Normalize
r² = x'² + y'²                             # Radial distance
d = 1 + k1*r² + k2*r⁴ + k3*r⁶             # Radial distortion
x'' = d*x' + 2*p1*x'*y' + p2*(r²+2*x'²)   # Tangential
y'' = d*y' + 2*p2*x'*y' + p1*(r²+2*y'²)
u = fx*x'' + cx, v = fy*y'' + cy          # Project to pixels
```

### Parameter Blocks

The factor supports optimization of:
- **Extrinsics (SE3 pose)**: 6 DOF (rotation + translation)
- **3D points**: 3 DOF per point
- **Intrinsics** (optional): 9 DOF (fx, fy, cx, cy, k1, k2, p1, p2, k3)

## WASM Interface Design

### JSON-Based API

The WASM interface uses JSON strings for data exchange:

```javascript
ba.set_cameras(JSON.stringify([...]));
ba.set_points(JSON.stringify([...]));
const result = JSON.parse(ba.optimize());
```

This was chosen over TypedArrays for:
- Simpler debugging (human-readable)
- Flexibility for varying data structures
- Acceptable performance for typical problem sizes

For very large problems, a TypedArray interface could be added.

### State Management

`WasmBundleAdjuster` maintains internal state:
- `cameras`: Vec of camera parameters
- `points`: Vec of 3D points
- `observations`: Vec of 2D observations
- `config`: Solver configuration

This allows incremental setup before calling `optimize()`.

## Testing Strategy

### Browser Tests (Playwright)

Tests run in real browsers to verify WASM execution:

```bash
npm test                 # Headless Chromium
npm run test:headed      # Visible browser
npm run test:ui          # Playwright UI
```

Test cases in `tests/bundle-adjustment.spec.ts`:
- Module initialization
- Single/multi-camera optimization
- Distortion handling
- Robust loss functions
- Convergence verification
- Edge cases (behind camera, etc.)

### Rust Unit Tests

Native Rust tests for the optimization logic:

```bash
cargo test
```

## Build Outputs

After `npm run build`:

| File | Size | Purpose |
|------|------|---------|
| `pkg/sba_solver_wasm.js` | ~15KB | JavaScript bindings |
| `pkg/sba_solver_wasm_bg.wasm` | ~720KB | WASM binary |
| `pkg/sba_solver_wasm.d.ts` | ~4KB | TypeScript definitions |

## Performance Considerations

### Bundle Size

The ~720KB WASM binary includes:
- Levenberg-Marquardt optimizer
- Sparse Cholesky factorization (faer)
- SE3 manifold operations
- All loss functions

### Runtime

Typical performance on modern hardware:
- Module load: ~30ms
- Small problem (10 cameras, 100 points): <10ms
- Medium problem (50 cameras, 1000 points): ~100ms
- Large problem (100+ cameras): scales with O(n) for sparse problems

### Optimization Tips

1. Use Huber loss for robustness without much overhead
2. Start with `optimize_points: false` to refine cameras first
3. Use `reference_camera_index` to fix gauge freedom
4. Filter outliers before optimization with `outlier_threshold`

## Updating apex-solver

To sync with upstream apex-solver:

1. Fetch upstream changes
2. Re-apply WASM compatibility patches:
   - Feature flags in `Cargo.toml`
   - `web-time` replacement for `std::time`
   - Conditional compilation for `rayon`/`memmap2`
3. Test with `cargo check --target wasm32-unknown-unknown`
4. Build and run browser tests

## References

- [apex-solver](https://github.com/amin-abouee/apex-solver) - Upstream optimization library
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) - Rust/JS interop
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) - Build tooling
- [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/) - Problem formulation
