# sba-solver-wasm

WebAssembly module for sparse bundle adjustment, enabling browser-based multicamera calibration refinement.

## Installation

### npm / CDN

```bash
npm install @talmolab/sba-solver-wasm
```

Or use directly from jsDelivr:

```javascript
import init, { WasmBundleAdjuster } from 'https://cdn.jsdelivr.net/npm/@talmolab/sba-solver-wasm@latest/sba_solver_wasm.js';
```

### Build from source

```bash
# Prerequisites: Rust nightly, wasm-pack
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# Build
npm run build

# Test
npm install
npm test
```

## Usage

```javascript
import init, { WasmBundleAdjuster } from '@talmolab/sba-solver-wasm';

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
  robust_loss: "huber",
  robust_loss_param: 1.0,
  optimize_extrinsics: true,
  optimize_points: true
}));

// Run optimization
const result = JSON.parse(ba.optimize());
console.log(`Converged: ${result.converged}, Final cost: ${result.final_cost}`);
```

## Features

- **Sparse Bundle Adjustment** - Levenberg-Marquardt optimization for camera calibration
- **Radial-Tangential Distortion** - Full Brown-Conrady model (k1, k2, k3, p1, p2)
- **SE3 Poses** - Proper manifold optimization for camera rotations
- **Robust Loss Functions** - Huber and Cauchy loss for outlier rejection
- **Pure WebAssembly** - No server required (~720KB)

## API Reference

### Camera Parameters

```typescript
interface CameraParams {
  rotation: [number, number, number, number];  // Quaternion [w, x, y, z]
  translation: [number, number, number];       // [x, y, z]
  focal: [number, number];                     // [fx, fy] in pixels
  principal: [number, number];                 // [cx, cy] in pixels
  distortion: [number, number, number, number, number];  // [k1, k2, p1, p2, k3]
}
```

### Observations

```typescript
interface Observation {
  camera_idx: number;  // Index into cameras array
  point_idx: number;   // Index into points array
  x: number;           // Observed x in pixels
  y: number;           // Observed y in pixels
}
```

### Solver Configuration

```typescript
interface SolverConfig {
  max_iterations?: number;       // Default: 100
  cost_tolerance?: number;       // Default: 1e-6
  parameter_tolerance?: number;  // Default: 1e-8
  gradient_tolerance?: number;   // Default: 1e-10
  robust_loss?: string;          // "none", "huber", or "cauchy"
  robust_loss_param?: number;    // Loss function parameter
  optimize_extrinsics?: boolean; // Default: true
  optimize_points?: boolean;     // Default: true
}
```

### Result

```typescript
interface BundleAdjustmentResult {
  cameras: CameraParams[];
  points: [number, number, number][];
  initial_cost: number;
  final_cost: number;
  iterations: number;
  converged: boolean;
  status: string;
}
```

## Development

```bash
# Run Rust tests
cargo test

# Build WASM (debug)
npm run build:dev

# Build WASM (release)
npm run build

# Run browser tests
npm test

# Run with visible browser
npm run test:headed

# Serve examples locally
npm run serve
# Open http://localhost:8080/examples/
```

## Technical Details

This module uses a fork of [apex-solver](https://github.com/amin-abouee/apex-solver) modified for WASM compatibility. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for architecture details, the apex-solver fork, and development notes.

## License

Apache-2.0
