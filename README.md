# sba-solver-wasm

A WebAssembly module for sparse bundle adjustment, enabling browser-based multicamera calibration refinement. Uses the Rust `apex-solver` crate internally for Levenberg-Marquardt optimization.

## Quick Start

```bash
# Build the WASM module
wasm-pack build --target web --release

# Run browser tests
npm install
npm test

# Or serve locally and open in browser
npm run serve
# Open http://localhost:8080/examples/
```

## Features

- **Sparse Bundle Adjustment** - Levenberg-Marquardt optimization for camera calibration
- **Radial-Tangential Distortion** - Full Brown-Conrady distortion model (k1, k2, k3, p1, p2)
- **SE3 Poses** - Proper manifold optimization for camera rotations
- **Robust Loss Functions** - Huber and Cauchy loss for outlier rejection
- **Pure WebAssembly** - No server-side computation required (~720KB)
- **Browser Tested** - Playwright test suite for Chrome, Firefox, and Safari

## JavaScript Usage

```javascript
import init, { WasmBundleAdjuster } from './pkg/sba_solver_wasm.js';

await init();
const ba = new WasmBundleAdjuster();

// Set camera parameters
ba.set_cameras(JSON.stringify([
  {
    rotation: [1.0, 0.0, 0.0, 0.0],    // Quaternion [w, x, y, z]
    translation: [0.0, 0.0, 0.0],      // [x, y, z]
    focal: [500.0, 500.0],             // [fx, fy]
    principal: [320.0, 240.0],         // [cx, cy]
    distortion: [0.0, 0.0, 0.0, 0.0, 0.0]  // [k1, k2, p1, p2, k3]
  }
]));

// Set 3D points
ba.set_points(JSON.stringify([
  [0.0, 0.0, 5.0],
  [1.0, 0.0, 5.0]
]));

// Set 2D observations
ba.set_observations(JSON.stringify([
  { camera_idx: 0, point_idx: 0, x: 320.0, y: 240.0 },
  { camera_idx: 0, point_idx: 1, x: 420.0, y: 240.0 }
]));

// Configure solver
ba.set_config(JSON.stringify({
  max_iterations: 100,
  cost_tolerance: 1e-6,
  robust_loss: "huber",
  robust_loss_param: 1.0,
  optimize_extrinsics: true,
  optimize_points: true
}));

// Run optimization
const result = JSON.parse(ba.optimize());
console.log(`Converged: ${result.converged}`);
console.log(`Final cost: ${result.final_cost}`);
console.log(`Optimized cameras:`, result.cameras);
console.log(`Optimized points:`, result.points);
```

## Data Formats

### Camera Parameters

```typescript
interface CameraParams {
  rotation: [number, number, number, number];  // Quaternion [w, x, y, z] (world-to-camera)
  translation: [number, number, number];       // Translation [x, y, z] (world-to-camera)
  focal: [number, number];                     // Focal lengths [fx, fy] in pixels
  principal: [number, number];                 // Principal point [cx, cy] in pixels
  distortion: [number, number, number, number, number];  // [k1, k2, p1, p2, k3]
}
```

### Observations

```typescript
interface Observation {
  camera_idx: number;  // Index into cameras array
  point_idx: number;   // Index into points array
  x: number;           // Observed x coordinate in pixels
  y: number;           // Observed y coordinate in pixels
}
```

### Solver Configuration

```typescript
interface SolverConfig {
  max_iterations: number;      // Maximum iterations (default: 100)
  cost_tolerance: number;      // Cost change tolerance (default: 1e-6)
  parameter_tolerance: number; // Parameter change tolerance (default: 1e-8)
  gradient_tolerance: number;  // Gradient tolerance (default: 1e-10)
  robust_loss: string;         // "none", "huber", or "cauchy"
  robust_loss_param: number;   // Loss function parameter
  optimize_extrinsics: boolean; // Optimize camera poses (default: true)
  optimize_points: boolean;     // Optimize 3D points (default: true)
}
```

### Result

```typescript
interface BundleAdjustmentResult {
  cameras: CameraParams[];     // Optimized camera parameters
  points: [number, number, number][];  // Optimized 3D points
  initial_cost: number;        // Initial sum of squared reprojection errors
  final_cost: number;          // Final cost after optimization
  iterations: number;          // Number of iterations performed
  converged: boolean;          // Whether the solver converged
  status: string;              // Convergence status message
}
```

## Project Structure

```
sba-solver-wasm/
├── Cargo.toml                    # Rust project config
├── package.json                  # Node.js config (for Playwright tests)
├── playwright.config.ts          # Playwright test configuration
├── .cargo/config.toml            # WASM build settings
├── .github/workflows/test.yml    # CI pipeline
├── apex-solver-fork/             # WASM-compatible fork of apex-solver
│   ├── Cargo.toml                # Modified with feature flags
│   └── src/                      # Feature-gated source
├── src/
│   └── lib.rs                    # WASM interface (~650 lines)
├── tests/
│   └── bundle-adjustment.spec.ts # Playwright browser tests (11 tests)
├── examples/
│   └── index.html                # Interactive browser demo
└── pkg/                          # Built WASM module (generated)
    ├── sba_solver_wasm.js        # JavaScript bindings
    ├── sba_solver_wasm.d.ts      # TypeScript types
    └── sba_solver_wasm_bg.wasm   # WASM binary (~720KB)
```

## Development

### Prerequisites

```bash
# Install Rust (nightly for edition 2024)
curl https://sh.rustup.rs -sSf | sh -s -- -y
source ~/.cargo/env

# Add WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
cargo install wasm-pack

# Install Node.js dependencies (for browser tests)
npm install
```

### Build Commands

```bash
# Check compilation (native)
cargo check

# Run Rust unit tests
cargo test

# Check WASM compilation
cargo check --target wasm32-unknown-unknown

# Build WASM module (release)
wasm-pack build --target web --release

# Build WASM module (debug, faster compilation)
wasm-pack build --target web --dev
```

### Browser Tests (Playwright)

```bash
# Install Playwright browsers (first time only)
npx playwright install

# Run all browser tests (Chromium)
npm test

# Run tests with visible browser
npm run test:headed

# Run tests with Playwright UI
npm run test:ui

# Run tests in debug mode
npm run test:debug
```

### Serving Locally

```bash
# Using npm serve (recommended)
npm run serve
# Open http://localhost:8080/examples/

# Or Python
python -m http.server 8080
```

## Technical Details

### apex-solver Integration

This module uses a fork of [apex-solver](https://github.com/amin-abouee/apex-solver), which provides:
- Levenberg-Marquardt optimization with adaptive damping
- Sparse Cholesky factorization via [faer](https://github.com/sarah-quinones/faer-rs)
- Manifold operations for SE3 poses (proper rotation handling)

### Custom ReprojectionFactor

We implement a custom `ReprojectionFactor` that:
1. Transforms 3D points from world to camera coordinates using SE3 pose
2. Projects points using pinhole model with radial-tangential distortion
3. Computes analytical Jacobians with respect to both pose (6 DOF) and point (3 DOF)

### WASM Compatibility

The apex-solver fork modifies the original crate for WASM compatibility:
- `memmap2`: Feature-gated as `io` (file I/O not needed in browser)
- `rayon`: Feature-gated as `parallel` (single-threaded in WASM)
- `std::time`: Replaced with `web-time` crate for WASM timing support
- Uses Rust 2024 edition for let-chain syntax
- `getrandom`: Configured with `wasm_js` backend

### Reprojection Model

The bundle adjustment minimizes reprojection error:

```
residual = project(R * point_world + t) - observed_2d
```

Where projection includes radial-tangential distortion:
```
x' = x/z, y' = y/z                    # Normalize
r² = x'² + y'²                        # Radial distance
d = 1 + k1*r² + k2*r⁴ + k3*r⁶        # Radial distortion
x'' = d*x' + 2*p1*x'*y' + p2*(r²+2*x'²)  # Apply distortion
y'' = d*y' + 2*p2*x'*y' + p1*(r²+2*y'²)
u = fx*x'' + cx, v = fy*y'' + cy      # Project to pixels
```

## CI/CD

GitHub Actions workflow (`.github/workflows/test.yml`) runs:
1. **Rust Tests** - Native unit tests with `cargo test`
2. **WASM Build** - Compile to WebAssembly with `wasm-pack`
3. **Playwright Tests** - Browser tests in Chromium, Firefox, and WebKit

## References

- [apex-solver GitHub](https://github.com/amin-abouee/apex-solver) - Optimization backend
- [apex-solver docs](https://docs.rs/apex-solver) - API documentation
- [wasm-bindgen guide](https://rustwasm.github.io/wasm-bindgen/) - WASM bindings
- [Playwright docs](https://playwright.dev/) - Browser testing
- [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/) - Problem reference

## License

Apache-2.0
