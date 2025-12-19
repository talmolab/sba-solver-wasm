# sba-solver-wasm - Task Tracking

## Project Status: RENAME IN PROGRESS

Package is being renamed from `apex-solver-wasm` to `sba-solver-wasm`.

## Remaining Tasks (After This Session)

### 1. Complete Rename (Manual Steps Required)

After quitting this session, run these commands:

```bash
# 1. Rename the folder
cd /home/talmo/code
mv apex-solver-wasm sba-solver-wasm
cd sba-solver-wasm

# 2. Rebuild WASM (this regenerates pkg/ with new names)
wasm-pack build --target web --release

# 3. Verify the build
ls pkg/
# Should show: sba_solver_wasm.js, sba_solver_wasm_bg.wasm, sba_solver_wasm.d.ts

# 4. Run tests to verify everything works
npm test

# 5. Update git remote (if pushing to new repo)
git remote set-url origin https://github.com/talmolab/sba-solver-wasm.git
```

### 2. Files Already Updated

- [x] `Cargo.toml` - package name, author, repo URL
- [x] `package.json` - name, description
- [x] `pkg/package.json` - name, author, repo URL, file references
- [x] `README.md` - all references updated
- [x] `examples/index.html` - title, heading, import path
- [x] `tests/bundle-adjustment.spec.ts` - import paths

### 3. Files NOT Updated (in scratch/ - gitignored)

These scratch files still reference the old name but are gitignored so don't need updating:
- `scratch/2025-12-17-sba-calibration-test/*.html` - old import paths
- `scratch/2025-12-18-packaging-investigation/*.js` - old import paths
- `scratch/2025-12-18-packaging-investigation/*.md` - old documentation

If you want to use the scratch test pages after rebuild, update their imports from:
```javascript
import init, { WasmBundleAdjuster } from '../../pkg/apex_solver_wasm.js';
```
to:
```javascript
import init, { WasmBundleAdjuster } from '../../pkg/sba_solver_wasm.js';
```

### 4. Integration with calibration-studio

After the rename and rebuild, copy files to calibration-studio:

```bash
# From sba-solver-wasm root
mkdir -p ../vibes/calibration-studio/lib/
cp pkg/sba_solver_wasm.js ../vibes/calibration-studio/lib/
cp pkg/sba_solver_wasm_bg.wasm ../vibes/calibration-studio/lib/
cp scratch/2025-12-18-packaging-investigation/sba-wrapper.js ../vibes/calibration-studio/lib/

# Update the wrapper's module path (edit lib/sba-wrapper.js line 28):
# Change: new URL('../../pkg/apex_solver_wasm.js', ...)
# To:     new URL('./sba_solver_wasm.js', ...)
```

Then follow `scratch/2025-12-18-packaging-investigation/IMPLEMENTATION_PROMPT.md` for integration steps.

---

## Completed Tasks

### 1. Investigation
- [x] Cloned apex-solver from https://github.com/amin-abouee/apex-solver
- [x] Documented API structure (Problem + LevenbergMarquardt pattern)
- [x] Identified WASM compatibility issues (memmap2, rayon, std::time)

### 2. Fork apex-solver for WASM
Location: `apex-solver-fork/`

**Changes made:**
- Feature flags: `io`, `parallel`, `cli`, `logging`, `visualization`
- Made `memmap2` and `rayon` optional dependencies
- Conditional compilation for parallel vs sequential iteration
- Replaced `std::time` with `web-time` crate for WASM timing
- Feature-gated `IoError` in error module
- Updated to Rust 2024 edition

### 3. Project Setup
- [x] Created `Cargo.toml` with WASM dependencies
- [x] Created `.cargo/config.toml` with SIMD128 and getrandom config
- [x] Configured `getrandom` with `wasm_js` backend

### 4. WASM Interface (`src/lib.rs`)
- [x] Data structures: `CameraParams`, `Observation`, `SolverConfig`, `BundleAdjustmentResult`
- [x] Custom `ReprojectionFactor` with analytical Jacobians
- [x] `WasmBundleAdjuster` class with JSON-based API
- [x] Huber and Cauchy robust loss support
- [x] SE3 manifold optimization for camera poses
- [x] **Intrinsics optimization** (focal length, principal point, distortion)
- [x] **Outlier filtering** (threshold-based observation filtering)
- [x] **Frame filtering** (ignore specific frames)
- [x] **Reference camera selection** (gauge fixing)

### 5. Build & Test
- [x] Native compilation passing (`cargo check`)
- [x] Rust unit tests passing (`cargo test`)
- [x] WASM compilation passing (`cargo check --target wasm32-unknown-unknown`)
- [x] WASM build successful (`wasm-pack build --target web --release`)
- [x] Output: ~720KB WASM binary

### 6. Browser Demo
- [x] Created `examples/index.html` with interactive UI
- [x] Synthetic data generation for testing
- [x] Configurable solver parameters
- [x] Real-time result display

### 7. Browser Testing (Playwright)
- [x] Created `tests/bundle-adjustment.spec.ts` (11 tests)
- [x] Created `playwright.config.ts` with multi-browser support
- [x] Created `package.json` with test scripts
- [x] All tests passing in Chromium

### 8. CI/CD
- [x] Created `.github/workflows/test.yml`
- [x] Rust tests job
- [x] WASM build job with artifact upload
- [x] Playwright browser tests job

### 9. Packaging Investigation
- [x] Tested various loading approaches (relative, absolute, dynamic import)
- [x] Created JS wrapper API (`sba-wrapper.js`)
- [x] Documented integration strategy for calibration-studio
- [x] Created implementation prompt for calibration-studio integration

### 10. Package Rename
- [x] Updated Cargo.toml
- [x] Updated package.json files
- [x] Updated README.md
- [x] Updated examples and tests
- [ ] Rebuild WASM (manual step after session)
- [ ] Rename folder (manual step after session)

## File Reference

| File | Purpose | Status |
|------|---------|--------|
| `Cargo.toml` | Rust project config | UPDATED |
| `package.json` | Node.js/Playwright config | UPDATED |
| `playwright.config.ts` | Browser test config | DONE |
| `.cargo/config.toml` | WASM build flags | DONE |
| `.github/workflows/test.yml` | CI pipeline | DONE |
| `apex-solver-fork/` | Fork with feature flags | DONE |
| `src/lib.rs` | WASM interface | DONE |
| `tests/bundle-adjustment.spec.ts` | Browser tests | UPDATED |
| `examples/index.html` | Browser demo | UPDATED |
| `pkg/` | Built WASM (needs rebuild) | NEEDS REBUILD |

## Technical Notes

### ReprojectionFactor Implementation

The custom `ReprojectionFactor` in `src/lib.rs`:
- Takes SE3 pose (6 DOF) and 3D point (3 DOF)
- Optionally takes intrinsics (9 DOF: fx, fy, cx, cy, k1, k2, p1, p2, k3)
- Transforms point from world to camera coordinates
- Projects using pinhole model with radial-tangential distortion
- Computes analytical Jacobians for all parameters

### WASM Compatibility Fixes

1. **`std::time::Instant`** - Not available in WASM
   - Solution: Added `web-time` crate as dependency

2. **`getrandom`** - Needs WASM backend configuration
   - Solution: Added `--cfg getrandom_backend="wasm_js"` to `.cargo/config.toml`

3. **`rayon`** - No multi-threading in WASM
   - Solution: Feature-gated with `#[cfg(feature = "parallel")]`

4. **`memmap2`** - No file system in browser
   - Solution: Feature-gated with `#[cfg(feature = "io")]`

## Future Improvements

### Potential Enhancements
- [ ] TypedArray input/output for better performance
- [ ] Progress callbacks during optimization
- [ ] Web Workers for non-blocking operation
- [ ] npm publishing for CDN distribution
