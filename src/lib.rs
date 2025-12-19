//! WASM module for sparse bundle adjustment using apex-solver.
//!
//! This module provides a WebAssembly interface for browser-based multicamera calibration.
//! It wraps the apex-solver crate's Levenberg-Marquardt optimizer for nonlinear least squares.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use apex_solver::core::loss_functions::{CauchyLoss, HuberLoss};
use apex_solver::core::problem::{Problem, VariableEnum};
use apex_solver::factors::Factor;
use apex_solver::manifold::ManifoldType;
use apex_solver::observers::OptObserver;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DMatrix, DVector, Matrix3, Quaternion, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[cfg(feature = "console_error_panic_hook")]
pub use console_error_panic_hook::set_once as set_panic_hook;

/// Initialize panic hook for better error messages in browser console.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// ============================================================================
// Data Structures
// ============================================================================

/// Camera parameters including extrinsics and intrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraParams {
    /// Rotation quaternion [w, x, y, z] (world-to-camera)
    pub rotation: [f64; 4],
    /// Translation vector [x, y, z] (world-to-camera)
    pub translation: [f64; 3],
    /// Focal lengths [fx, fy]
    pub focal: [f64; 2],
    /// Principal point [cx, cy]
    pub principal: [f64; 2],
    /// Distortion coefficients [k1, k2, p1, p2, k3] (radial-tangential)
    pub distortion: [f64; 5],
}

impl CameraParams {
    /// Create camera intrinsics vector for apex-solver: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    pub fn intrinsics_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.focal[0],
            self.focal[1],
            self.principal[0],
            self.principal[1],
            self.distortion[0], // k1
            self.distortion[1], // k2
            self.distortion[2], // p1
            self.distortion[3], // p2
            self.distortion[4], // k3
        ])
    }

    /// Get rotation as UnitQuaternion
    pub fn rotation_quat(&self) -> UnitQuaternion<f64> {
        UnitQuaternion::from_quaternion(Quaternion::new(
            self.rotation[0], // w
            self.rotation[1], // x
            self.rotation[2], // y
            self.rotation[3], // z
        ))
    }

    /// Get translation as Vector3
    pub fn translation_vec(&self) -> Vector3<f64> {
        Vector3::new(
            self.translation[0],
            self.translation[1],
            self.translation[2],
        )
    }

    /// Transform a 3D point from world coordinates to camera coordinates.
    pub fn world_to_camera(&self, point_world: &Vector3<f64>) -> Vector3<f64> {
        let r = self.rotation_quat();
        let t = self.translation_vec();
        r * point_world + t
    }

    /// Project a 3D world point to 2D pixel coordinates using this camera.
    /// Returns None if the point is behind the camera.
    pub fn project(&self, point_world: &Vector3<f64>) -> Option<(f64, f64)> {
        let point_cam = self.world_to_camera(point_world);

        let x = point_cam[0];
        let y = point_cam[1];
        let z = point_cam[2];

        // Check if point is behind camera
        if z < f64::EPSILON.sqrt() {
            return None;
        }

        // Normalized image coordinates
        let x_prime = x / z;
        let y_prime = y / z;

        // Intrinsics
        let fx = self.focal[0];
        let fy = self.focal[1];
        let cx = self.principal[0];
        let cy = self.principal[1];
        let k1 = self.distortion[0];
        let k2 = self.distortion[1];
        let p1 = self.distortion[2];
        let p2 = self.distortion[3];
        let k3 = self.distortion[4];

        // Radial distortion
        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let d = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        // Apply distortion
        let x_distorted =
            d * x_prime + 2.0 * p1 * x_prime * y_prime + p2 * (r2 + 2.0 * x_prime * x_prime);
        let y_distorted =
            d * y_prime + 2.0 * p2 * x_prime * y_prime + p1 * (r2 + 2.0 * y_prime * y_prime);

        // Project to pixel coordinates
        let u = fx * x_distorted + cx;
        let v = fy * y_distorted + cy;

        Some((u, v))
    }

    /// Compute reprojection error for a 3D point and observed 2D point.
    /// Returns the Euclidean distance in pixels, or None if the point is behind the camera.
    pub fn reprojection_error(
        &self,
        point_world: &Vector3<f64>,
        observed: (f64, f64),
    ) -> Option<f64> {
        let (u, v) = self.project(point_world)?;
        let dx = u - observed.0;
        let dy = v - observed.1;
        Some((dx * dx + dy * dy).sqrt())
    }
}

/// A single 2D observation of a 3D point in a camera image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Index of the camera that made this observation
    pub camera_idx: usize,
    /// Index of the 3D point being observed
    pub point_idx: usize,
    /// Observed x coordinate in image (pixels)
    pub x: f64,
    /// Observed y coordinate in image (pixels)
    pub y: f64,
}

/// Configuration for the bundle adjustment solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Maximum number of iterations (default: 100)
    pub max_iterations: usize,
    /// Cost change tolerance for convergence (default: 1e-6)
    pub cost_tolerance: f64,
    /// Parameter change tolerance for convergence (default: 1e-8)
    pub parameter_tolerance: f64,
    /// Gradient tolerance for convergence (default: 1e-10)
    pub gradient_tolerance: f64,
    /// Robust loss function: "none", "huber", or "cauchy"
    pub robust_loss: String,
    /// Parameter for robust loss (Huber delta or Cauchy scale)
    pub robust_loss_param: f64,
    /// Whether to optimize camera extrinsics (poses)
    pub optimize_extrinsics: bool,
    /// Whether to optimize 3D points
    pub optimize_points: bool,
    /// Whether to optimize camera intrinsics (focal length, principal point, distortion)
    #[serde(default)]
    pub optimize_intrinsics: bool,
    /// Outlier threshold in pixels. Observations with initial reprojection error > threshold are excluded.
    /// Set to 0 or negative to disable outlier filtering (default: 0 = disabled)
    #[serde(default)]
    pub outlier_threshold: f64,
    /// Index of camera to fix as reference (gauge freedom). Default: 0 (first camera)
    #[serde(default)]
    pub reference_camera: usize,
    /// Frame indices to ignore during optimization. Points belonging to these frames will be excluded.
    /// Requires point_to_frame mapping to be set. Empty array = use all frames.
    #[serde(default)]
    pub ignore_frames: Vec<usize>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            cost_tolerance: 1e-6,
            parameter_tolerance: 1e-8,
            gradient_tolerance: 1e-10,
            robust_loss: "huber".to_string(),
            robust_loss_param: 1.0,
            optimize_extrinsics: true,
            optimize_points: true,
            optimize_intrinsics: false,
            outlier_threshold: 0.0,    // disabled by default
            reference_camera: 0,       // first camera is reference by default
            ignore_frames: Vec::new(), // no frames ignored by default
        }
    }
}

/// Result of bundle adjustment optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleAdjustmentResult {
    /// Optimized camera parameters
    pub cameras: Vec<CameraParams>,
    /// Optimized 3D point positions
    pub points: Vec<[f64; 3]>,
    /// Initial cost (sum of squared reprojection errors)
    pub initial_cost: f64,
    /// Final cost after optimization
    pub final_cost: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the solver converged
    pub converged: bool,
    /// Status message
    pub status: String,
    /// Cost at each iteration (for loss curve visualization)
    pub cost_history: Vec<f64>,
    /// Number of observations used (after all filtering)
    pub num_observations_used: usize,
    /// Number of observations filtered as outliers (by reprojection error threshold)
    pub num_observations_filtered: usize,
    /// Number of observations filtered due to ignored frames
    pub num_observations_filtered_by_frame: usize,
}

// ============================================================================
// Cost History Observer
// ============================================================================

/// Observer that collects cost at each iteration for loss curve visualization.
struct CostHistoryObserver {
    costs: Arc<Mutex<Vec<f64>>>,
}

impl CostHistoryObserver {
    fn new() -> (Self, Arc<Mutex<Vec<f64>>>) {
        let costs = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                costs: costs.clone(),
            },
            costs,
        )
    }
}

impl OptObserver for CostHistoryObserver {
    fn on_step(&self, _values: &HashMap<String, VariableEnum>, _iteration: usize) {
        // Cost is set via set_iteration_metrics
    }

    fn set_iteration_metrics(
        &self,
        cost: f64,
        _gradient_norm: f64,
        _damping: Option<f64>,
        _step_norm: f64,
        _step_quality: Option<f64>,
    ) {
        if let Ok(mut guard) = self.costs.lock() {
            guard.push(cost);
        }
    }
}

// ============================================================================
// Custom Factor for Bundle Adjustment
// ============================================================================

/// Reprojection factor for bundle adjustment.
///
/// This factor computes the reprojection error for a single observation:
/// - Transforms a 3D point from world coordinates to camera coordinates using SE3 pose
/// - Projects the point using radial-tangential distortion model
/// - Computes residual as (projected - observed) pixel coordinates
///
/// Variables:
/// - Camera pose (SE3): [tx, ty, tz, qw, qx, qy, qz]
/// - 3D point (R3): [x, y, z]
///
/// The Jacobian is computed analytically with respect to both variables.
#[derive(Debug, Clone)]
pub struct ReprojectionFactor {
    /// Observed 2D point [u, v]
    pub observed: [f64; 2],
    /// Camera intrinsics [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    pub intrinsics: DVector<f64>,
}

impl ReprojectionFactor {
    pub fn new(observed: [f64; 2], intrinsics: DVector<f64>) -> Self {
        Self {
            observed,
            intrinsics,
        }
    }

    /// Project a 3D point in camera coordinates to 2D using radial-tangential model.
    fn project(&self, point_cam: &Vector3<f64>) -> Option<(f64, f64)> {
        let x = point_cam[0];
        let y = point_cam[1];
        let z = point_cam[2];

        // Check if point is behind camera
        if z < f64::EPSILON.sqrt() {
            return None;
        }

        // Normalized image coordinates
        let x_prime = x / z;
        let y_prime = y / z;

        // Distortion coefficients
        let fx = self.intrinsics[0];
        let fy = self.intrinsics[1];
        let cx = self.intrinsics[2];
        let cy = self.intrinsics[3];
        let k1 = self.intrinsics[4];
        let k2 = self.intrinsics[5];
        let p1 = self.intrinsics[6];
        let p2 = self.intrinsics[7];
        let k3 = self.intrinsics[8];

        // Radial distortion
        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let d = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        // Apply distortion
        let x_distorted =
            d * x_prime + 2.0 * p1 * x_prime * y_prime + p2 * (r2 + 2.0 * x_prime * x_prime);
        let y_distorted =
            d * y_prime + 2.0 * p2 * x_prime * y_prime + p1 * (r2 + 2.0 * y_prime * y_prime);

        // Project to pixel coordinates
        let u = fx * x_distorted + cx;
        let v = fy * y_distorted + cy;

        Some((u, v))
    }

    /// Compute Jacobian of projection with respect to 3D point in camera coordinates.
    /// Returns a 2x3 matrix: d(u,v)/d(x,y,z)
    fn projection_jacobian_point(&self, point_cam: &Vector3<f64>) -> Option<DMatrix<f64>> {
        let x = point_cam[0];
        let y = point_cam[1];
        let z = point_cam[2];

        if z < f64::EPSILON.sqrt() {
            return None;
        }

        let fx = self.intrinsics[0];
        let fy = self.intrinsics[1];
        let k1 = self.intrinsics[4];
        let k2 = self.intrinsics[5];
        let p1 = self.intrinsics[6];
        let p2 = self.intrinsics[7];
        let k3 = self.intrinsics[8];

        let x_prime = x / z;
        let y_prime = y / z;
        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;

        let d = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2;
        let d_d_r2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

        // Derivatives of normalized coords w.r.t. 3D point
        let dx_prime_dx = 1.0 / z;
        let dx_prime_dz = -x / (z * z);
        let dy_prime_dy = 1.0 / z;
        let dy_prime_dz = -y / (z * z);

        // Derivatives of r2 w.r.t. normalized coords
        let dr2_dx_prime = 2.0 * x_prime;
        let dr2_dy_prime = 2.0 * y_prime;

        // Derivatives of distorted coords w.r.t. normalized coords
        let dx_dist_dx_prime = d
            + x_prime * d_d_r2 * dr2_dx_prime
            + 2.0 * p1 * y_prime
            + p2 * (dr2_dx_prime + 4.0 * x_prime);
        let dx_dist_dy_prime =
            x_prime * d_d_r2 * dr2_dy_prime + 2.0 * p1 * x_prime + p2 * dr2_dy_prime;
        let dy_dist_dx_prime =
            y_prime * d_d_r2 * dr2_dx_prime + 2.0 * p2 * y_prime + p1 * dr2_dx_prime;
        let dy_dist_dy_prime = d
            + y_prime * d_d_r2 * dr2_dy_prime
            + 2.0 * p2 * x_prime
            + p1 * (dr2_dy_prime + 4.0 * y_prime);

        // Chain rule: d(u,v)/d(x,y,z)
        let mut jac = DMatrix::zeros(2, 3);

        // du/dx = fx * dx_dist/dx_prime * dx_prime/dx
        jac[(0, 0)] = fx * dx_dist_dx_prime * dx_prime_dx;
        // du/dy = fx * dx_dist/dy_prime * dy_prime/dy
        jac[(0, 1)] = fx * dx_dist_dy_prime * dy_prime_dy;
        // du/dz = fx * (dx_dist/dx_prime * dx_prime/dz + dx_dist/dy_prime * dy_prime/dz)
        jac[(0, 2)] = fx * (dx_dist_dx_prime * dx_prime_dz + dx_dist_dy_prime * dy_prime_dz);

        // dv/dx = fy * dy_dist/dx_prime * dx_prime/dx
        jac[(1, 0)] = fy * dy_dist_dx_prime * dx_prime_dx;
        // dv/dy = fy * dy_dist/dy_prime * dy_prime/dy
        jac[(1, 1)] = fy * dy_dist_dy_prime * dy_prime_dy;
        // dv/dz = fy * (dy_dist/dx_prime * dx_prime/dz + dy_dist/dy_prime * dy_prime/dz)
        jac[(1, 2)] = fy * (dy_dist_dx_prime * dx_prime_dz + dy_dist_dy_prime * dy_prime_dz);

        Some(jac)
    }
}

impl Factor for ReprojectionFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // params[0] = camera pose SE3: [tx, ty, tz, qw, qx, qy, qz]
        // params[1] = 3D point: [x, y, z]
        let pose = &params[0];
        let point = &params[1];

        // Extract pose components
        let t = Vector3::new(pose[0], pose[1], pose[2]);
        let q =
            UnitQuaternion::from_quaternion(Quaternion::new(pose[3], pose[4], pose[5], pose[6]));

        // Extract 3D point
        let point_world = Vector3::new(point[0], point[1], point[2]);

        // Transform point to camera coordinates
        let point_cam = q * point_world + t;

        // Project to 2D
        match self.project(&point_cam) {
            Some((u, v)) => {
                // Compute residual
                let residual = DVector::from_vec(vec![u - self.observed[0], v - self.observed[1]]);

                let jacobian = if compute_jacobian {
                    // Compute Jacobian
                    // Total Jacobian is: [d_residual/d_pose (2x6), d_residual/d_point (2x3)]
                    //
                    // d_residual/d_point = d_proj/d_point_cam * d_point_cam/d_point_world
                    // d_point_cam/d_point_world = R (rotation matrix)
                    //
                    // d_residual/d_pose is more complex due to SE3 manifold

                    let d_proj_d_point_cam = self.projection_jacobian_point(&point_cam);

                    match d_proj_d_point_cam {
                        Some(d_proj) => {
                            // d_point_cam/d_point_world = R
                            let r_mat = q.to_rotation_matrix();
                            let d_point_cam_d_point = r_mat.matrix();

                            // Jacobian w.r.t. 3D point: 2x3
                            let jac_point = &d_proj * d_point_cam_d_point;

                            // Jacobian w.r.t. pose (SE3 tangent space: 6 DOF)
                            // SE3 tangent: [v_x, v_y, v_z, omega_x, omega_y, omega_z]
                            // where v is translation perturbation and omega is rotation perturbation
                            //
                            // For the transformation point_cam = R * point_world + t:
                            // - Translation perturbation: d_point_cam/d_v = R (rotation matrix)
                            // - Rotation perturbation: d_point_cam/d_omega = -[point_cam]_x (skew symmetric)

                            let mut jac_pose = DMatrix::zeros(2, 6);

                            // Translation part: d_proj/d_point_cam * d_point_cam/d_v = d_proj/d_point_cam * R
                            let jac_translation = &d_proj * r_mat.matrix();
                            for i in 0..2 {
                                for j in 0..3 {
                                    jac_pose[(i, j)] = jac_translation[(i, j)];
                                }
                            }

                            // Rotation part: d_proj/d_point_cam * d_point_cam/d_omega
                            // d_point_cam/d_omega = -skew(point_cam)
                            let skew = Matrix3::new(
                                0.0,
                                -point_cam[2],
                                point_cam[1],
                                point_cam[2],
                                0.0,
                                -point_cam[0],
                                -point_cam[1],
                                point_cam[0],
                                0.0,
                            );
                            let d_point_cam_d_omega = -skew;
                            let jac_rotation = &d_proj * d_point_cam_d_omega;

                            for i in 0..2 {
                                for j in 0..3 {
                                    jac_pose[(i, 3 + j)] = jac_rotation[(i, j)];
                                }
                            }

                            // Combine: [jac_pose (2x6), jac_point (2x3)]
                            let mut full_jac = DMatrix::zeros(2, 9);
                            for i in 0..2 {
                                for j in 0..6 {
                                    full_jac[(i, j)] = jac_pose[(i, j)];
                                }
                                for j in 0..3 {
                                    full_jac[(i, 6 + j)] = jac_point[(i, j)];
                                }
                            }

                            Some(full_jac)
                        }
                        None => {
                            // Invalid projection, return zero Jacobian
                            Some(DMatrix::zeros(2, 9))
                        }
                    }
                } else {
                    None
                };

                (residual, jacobian)
            }
            None => {
                // Point is behind camera - return large residual
                let residual = DVector::from_vec(vec![1e6, 1e6]);
                let jacobian = if compute_jacobian {
                    Some(DMatrix::zeros(2, 9))
                } else {
                    None
                };
                (residual, jacobian)
            }
        }
    }

    fn get_dimension(&self) -> usize {
        2 // (u, v) residual
    }
}

// ============================================================================
// Reprojection Factor with Intrinsics Optimization
// ============================================================================

/// Reprojection factor that also optimizes camera intrinsics.
///
/// Variables:
/// - Camera pose (SE3): [tx, ty, tz, qw, qx, qy, qz] - 7 params (6 DOF)
/// - 3D point (R3): [x, y, z] - 3 params
/// - Camera intrinsics (R9): [fx, fy, cx, cy, k1, k2, p1, p2, k3] - 9 params
#[derive(Debug, Clone)]
pub struct ReprojectionFactorWithIntrinsics {
    /// Observed 2D point [u, v]
    pub observed: [f64; 2],
}

impl ReprojectionFactorWithIntrinsics {
    pub fn new(observed: [f64; 2]) -> Self {
        Self { observed }
    }

    /// Project a 3D point in camera coordinates to 2D using radial-tangential model.
    fn project(point_cam: &Vector3<f64>, intrinsics: &DVector<f64>) -> Option<(f64, f64)> {
        let x = point_cam[0];
        let y = point_cam[1];
        let z = point_cam[2];

        if z < f64::EPSILON.sqrt() {
            return None;
        }

        let x_prime = x / z;
        let y_prime = y / z;

        let fx = intrinsics[0];
        let fy = intrinsics[1];
        let cx = intrinsics[2];
        let cy = intrinsics[3];
        let k1 = intrinsics[4];
        let k2 = intrinsics[5];
        let p1 = intrinsics[6];
        let p2 = intrinsics[7];
        let k3 = intrinsics[8];

        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let d = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        let x_distorted =
            d * x_prime + 2.0 * p1 * x_prime * y_prime + p2 * (r2 + 2.0 * x_prime * x_prime);
        let y_distorted =
            d * y_prime + 2.0 * p2 * x_prime * y_prime + p1 * (r2 + 2.0 * y_prime * y_prime);

        let u = fx * x_distorted + cx;
        let v = fy * y_distorted + cy;

        Some((u, v))
    }

    /// Compute Jacobians w.r.t. point in camera coordinates, and intrinsics.
    /// Returns (d_proj/d_point_cam (2x3), d_proj/d_intrinsics (2x9))
    fn projection_jacobians(
        point_cam: &Vector3<f64>,
        intrinsics: &DVector<f64>,
    ) -> Option<(DMatrix<f64>, DMatrix<f64>)> {
        let x = point_cam[0];
        let y = point_cam[1];
        let z = point_cam[2];

        if z < f64::EPSILON.sqrt() {
            return None;
        }

        let fx = intrinsics[0];
        let fy = intrinsics[1];
        let k1 = intrinsics[4];
        let k2 = intrinsics[5];
        let p1 = intrinsics[6];
        let p2 = intrinsics[7];
        let k3 = intrinsics[8];

        let x_prime = x / z;
        let y_prime = y / z;
        let r2 = x_prime * x_prime + y_prime * y_prime;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let d = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let d_d_r2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

        // Distorted coordinates
        let x_distorted =
            d * x_prime + 2.0 * p1 * x_prime * y_prime + p2 * (r2 + 2.0 * x_prime * x_prime);
        let y_distorted =
            d * y_prime + 2.0 * p2 * x_prime * y_prime + p1 * (r2 + 2.0 * y_prime * y_prime);

        // Derivatives of normalized coords w.r.t. 3D point
        let dx_prime_dx = 1.0 / z;
        let dx_prime_dz = -x / (z * z);
        let dy_prime_dy = 1.0 / z;
        let dy_prime_dz = -y / (z * z);

        // Derivatives of r2 w.r.t. normalized coords
        let dr2_dx_prime = 2.0 * x_prime;
        let dr2_dy_prime = 2.0 * y_prime;

        // Derivatives of distorted coords w.r.t. normalized coords
        let dx_dist_dx_prime = d
            + x_prime * d_d_r2 * dr2_dx_prime
            + 2.0 * p1 * y_prime
            + p2 * (dr2_dx_prime + 4.0 * x_prime);
        let dx_dist_dy_prime =
            x_prime * d_d_r2 * dr2_dy_prime + 2.0 * p1 * x_prime + p2 * dr2_dy_prime;
        let dy_dist_dx_prime =
            y_prime * d_d_r2 * dr2_dx_prime + 2.0 * p2 * y_prime + p1 * dr2_dx_prime;
        let dy_dist_dy_prime = d
            + y_prime * d_d_r2 * dr2_dy_prime
            + 2.0 * p2 * x_prime
            + p1 * (dr2_dy_prime + 4.0 * y_prime);

        // Jacobian w.r.t. point in camera coordinates (2x3)
        let mut jac_point = DMatrix::zeros(2, 3);
        jac_point[(0, 0)] = fx * dx_dist_dx_prime * dx_prime_dx;
        jac_point[(0, 1)] = fx * dx_dist_dy_prime * dy_prime_dy;
        jac_point[(0, 2)] = fx * (dx_dist_dx_prime * dx_prime_dz + dx_dist_dy_prime * dy_prime_dz);
        jac_point[(1, 0)] = fy * dy_dist_dx_prime * dx_prime_dx;
        jac_point[(1, 1)] = fy * dy_dist_dy_prime * dy_prime_dy;
        jac_point[(1, 2)] = fy * (dy_dist_dx_prime * dx_prime_dz + dy_dist_dy_prime * dy_prime_dz);

        // Jacobian w.r.t. intrinsics (2x9)
        // Intrinsics: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        let mut jac_intrinsics = DMatrix::zeros(2, 9);

        // du/dfx = x_distorted
        jac_intrinsics[(0, 0)] = x_distorted;
        // du/dfy = 0
        jac_intrinsics[(0, 1)] = 0.0;
        // du/dcx = 1
        jac_intrinsics[(0, 2)] = 1.0;
        // du/dcy = 0
        jac_intrinsics[(0, 3)] = 0.0;
        // du/dk1 = fx * x_prime * r2
        jac_intrinsics[(0, 4)] = fx * x_prime * r2;
        // du/dk2 = fx * x_prime * r4
        jac_intrinsics[(0, 5)] = fx * x_prime * r4;
        // du/dp1 = fx * 2 * x_prime * y_prime
        jac_intrinsics[(0, 6)] = fx * 2.0 * x_prime * y_prime;
        // du/dp2 = fx * (r2 + 2 * x_prime^2)
        jac_intrinsics[(0, 7)] = fx * (r2 + 2.0 * x_prime * x_prime);
        // du/dk3 = fx * x_prime * r6
        jac_intrinsics[(0, 8)] = fx * x_prime * r6;

        // dv/dfx = 0
        jac_intrinsics[(1, 0)] = 0.0;
        // dv/dfy = y_distorted
        jac_intrinsics[(1, 1)] = y_distorted;
        // dv/dcx = 0
        jac_intrinsics[(1, 2)] = 0.0;
        // dv/dcy = 1
        jac_intrinsics[(1, 3)] = 1.0;
        // dv/dk1 = fy * y_prime * r2
        jac_intrinsics[(1, 4)] = fy * y_prime * r2;
        // dv/dk2 = fy * y_prime * r4
        jac_intrinsics[(1, 5)] = fy * y_prime * r4;
        // dv/dp1 = fy * (r2 + 2 * y_prime^2)
        jac_intrinsics[(1, 6)] = fy * (r2 + 2.0 * y_prime * y_prime);
        // dv/dp2 = fy * 2 * x_prime * y_prime
        jac_intrinsics[(1, 7)] = fy * 2.0 * x_prime * y_prime;
        // dv/dk3 = fy * y_prime * r6
        jac_intrinsics[(1, 8)] = fy * y_prime * r6;

        Some((jac_point, jac_intrinsics))
    }
}

impl Factor for ReprojectionFactorWithIntrinsics {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // params[0] = camera pose SE3: [tx, ty, tz, qw, qx, qy, qz]
        // params[1] = 3D point: [x, y, z]
        // params[2] = intrinsics: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        let pose = &params[0];
        let point = &params[1];
        let intrinsics = &params[2];

        let t = Vector3::new(pose[0], pose[1], pose[2]);
        let q =
            UnitQuaternion::from_quaternion(Quaternion::new(pose[3], pose[4], pose[5], pose[6]));

        let point_world = Vector3::new(point[0], point[1], point[2]);
        let point_cam = q * point_world + t;

        match Self::project(&point_cam, intrinsics) {
            Some((u, v)) => {
                let residual = DVector::from_vec(vec![u - self.observed[0], v - self.observed[1]]);

                let jacobian = if compute_jacobian {
                    match Self::projection_jacobians(&point_cam, intrinsics) {
                        Some((d_proj_d_point_cam, jac_intrinsics)) => {
                            let r_mat = q.to_rotation_matrix();

                            // Jacobian w.r.t. 3D point: 2x3
                            let jac_point = &d_proj_d_point_cam * r_mat.matrix();

                            // Jacobian w.r.t. pose (SE3 tangent space: 6 DOF)
                            let mut jac_pose = DMatrix::zeros(2, 6);

                            // Translation part
                            let jac_translation = &d_proj_d_point_cam * r_mat.matrix();
                            for i in 0..2 {
                                for j in 0..3 {
                                    jac_pose[(i, j)] = jac_translation[(i, j)];
                                }
                            }

                            // Rotation part
                            let skew = Matrix3::new(
                                0.0,
                                -point_cam[2],
                                point_cam[1],
                                point_cam[2],
                                0.0,
                                -point_cam[0],
                                -point_cam[1],
                                point_cam[0],
                                0.0,
                            );
                            let d_point_cam_d_omega = -skew;
                            let jac_rotation = &d_proj_d_point_cam * d_point_cam_d_omega;

                            for i in 0..2 {
                                for j in 0..3 {
                                    jac_pose[(i, 3 + j)] = jac_rotation[(i, j)];
                                }
                            }

                            // Combine: [jac_pose (2x6), jac_point (2x3), jac_intrinsics (2x9)]
                            let mut full_jac = DMatrix::zeros(2, 18);
                            for i in 0..2 {
                                for j in 0..6 {
                                    full_jac[(i, j)] = jac_pose[(i, j)];
                                }
                                for j in 0..3 {
                                    full_jac[(i, 6 + j)] = jac_point[(i, j)];
                                }
                                for j in 0..9 {
                                    full_jac[(i, 9 + j)] = jac_intrinsics[(i, j)];
                                }
                            }

                            Some(full_jac)
                        }
                        None => Some(DMatrix::zeros(2, 18)),
                    }
                } else {
                    None
                };

                (residual, jacobian)
            }
            None => {
                let residual = DVector::from_vec(vec![1e6, 1e6]);
                let jacobian = if compute_jacobian {
                    Some(DMatrix::zeros(2, 18))
                } else {
                    None
                };
                (residual, jacobian)
            }
        }
    }

    fn get_dimension(&self) -> usize {
        2
    }
}

// ============================================================================
// WASM Interface
// ============================================================================

/// WebAssembly interface for bundle adjustment.
#[wasm_bindgen]
pub struct WasmBundleAdjuster {
    cameras: Vec<CameraParams>,
    points: Vec<[f64; 3]>,
    observations: Vec<Observation>,
    config: SolverConfig,
    /// Mapping from point index to frame index (for frame filtering)
    point_to_frame: Vec<usize>,
}

#[wasm_bindgen]
impl WasmBundleAdjuster {
    /// Create a new bundle adjuster instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            cameras: Vec::new(),
            points: Vec::new(),
            observations: Vec::new(),
            config: SolverConfig::default(),
            point_to_frame: Vec::new(),
        }
    }

    /// Set cameras from JSON string.
    #[wasm_bindgen]
    pub fn set_cameras(&mut self, cameras_json: &str) -> Result<(), JsValue> {
        self.cameras = serde_json::from_str(cameras_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse cameras: {}", e)))?;
        Ok(())
    }

    /// Set 3D points from JSON string.
    #[wasm_bindgen]
    pub fn set_points(&mut self, points_json: &str) -> Result<(), JsValue> {
        self.points = serde_json::from_str(points_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse points: {}", e)))?;
        Ok(())
    }

    /// Set observations from JSON string.
    #[wasm_bindgen]
    pub fn set_observations(&mut self, observations_json: &str) -> Result<(), JsValue> {
        self.observations = serde_json::from_str(observations_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse observations: {}", e)))?;
        Ok(())
    }

    /// Set solver configuration from JSON string.
    #[wasm_bindgen]
    pub fn set_config(&mut self, config_json: &str) -> Result<(), JsValue> {
        self.config = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse config: {}", e)))?;
        Ok(())
    }

    /// Set point-to-frame mapping from JSON string.
    /// This is an array where point_to_frame[i] is the frame index for point i.
    /// Required for frame filtering (ignore_frames config option).
    #[wasm_bindgen]
    pub fn set_point_to_frame(&mut self, point_to_frame_json: &str) -> Result<(), JsValue> {
        self.point_to_frame = serde_json::from_str(point_to_frame_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse point_to_frame: {}", e)))?;
        Ok(())
    }

    /// Get current configuration as JSON.
    #[wasm_bindgen]
    pub fn get_config(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.config)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize config: {}", e)))
    }

    /// Run bundle adjustment optimization.
    /// Returns a JSON string with the optimization result.
    #[wasm_bindgen]
    pub fn optimize(&self) -> Result<String, JsValue> {
        // Validate inputs
        if self.cameras.is_empty() {
            return Err(JsValue::from_str("No cameras provided"));
        }
        if self.points.is_empty() {
            return Err(JsValue::from_str("No points provided"));
        }
        if self.observations.is_empty() {
            return Err(JsValue::from_str("No observations provided"));
        }

        // Build optimization problem
        let mut problem = Problem::new();
        let mut initial_values: HashMap<String, (ManifoldType, DVector<f64>)> = HashMap::new();

        // Add camera pose variables (SE3)
        for (i, cam) in self.cameras.iter().enumerate() {
            let var_name = format!("cam{}", i);
            // SE3 format: [tx, ty, tz, qw, qx, qy, qz]
            let pose_data = DVector::from_vec(vec![
                cam.translation[0],
                cam.translation[1],
                cam.translation[2],
                cam.rotation[0], // qw
                cam.rotation[1], // qx
                cam.rotation[2], // qy
                cam.rotation[3], // qz
            ]);
            initial_values.insert(var_name, (ManifoldType::SE3, pose_data));
        }

        // Add 3D point variables (Euclidean R3)
        for (i, pt) in self.points.iter().enumerate() {
            let var_name = format!("pt{}", i);
            let point_data = DVector::from_vec(vec![pt[0], pt[1], pt[2]]);
            initial_values.insert(var_name, (ManifoldType::RN, point_data));
        }

        // Add camera intrinsics variables if optimizing intrinsics
        if self.config.optimize_intrinsics {
            for (i, cam) in self.cameras.iter().enumerate() {
                let var_name = format!("intr{}", i);
                let intrinsics_data = cam.intrinsics_vector();
                initial_values.insert(var_name, (ManifoldType::RN, intrinsics_data));
            }
        }

        // Add reprojection factors for each observation (with optional filtering)
        let outlier_filtering_enabled = self.config.outlier_threshold > 0.0;
        let frame_filtering_enabled =
            !self.config.ignore_frames.is_empty() && !self.point_to_frame.is_empty();

        // Build a HashSet of ignored frames for O(1) lookup
        let ignored_frames: std::collections::HashSet<usize> =
            self.config.ignore_frames.iter().cloned().collect();

        let mut num_observations_used = 0usize;
        let mut num_observations_filtered = 0usize;
        let mut num_observations_filtered_by_frame = 0usize;

        for obs in &self.observations {
            // Frame filtering: skip observations whose points belong to ignored frames
            if frame_filtering_enabled {
                if let Some(&frame) = self.point_to_frame.get(obs.point_idx) {
                    if ignored_frames.contains(&frame) {
                        num_observations_filtered_by_frame += 1;
                        continue; // Skip this observation
                    }
                }
            }

            // Compute initial reprojection error for outlier filtering
            if outlier_filtering_enabled {
                let cam = &self.cameras[obs.camera_idx];
                let pt = &self.points[obs.point_idx];
                let point_world = Vector3::new(pt[0], pt[1], pt[2]);

                if let Some(error) = cam.reprojection_error(&point_world, (obs.x, obs.y)) {
                    if error > self.config.outlier_threshold {
                        num_observations_filtered += 1;
                        continue; // Skip this observation
                    }
                }
            }

            num_observations_used += 1;
            let cam_var = format!("cam{}", obs.camera_idx);
            let pt_var = format!("pt{}", obs.point_idx);

            // Create robust loss function
            let loss: Option<Box<dyn apex_solver::core::loss_functions::LossFunction + Send>> =
                match self.config.robust_loss.as_str() {
                    "huber" => HuberLoss::new(self.config.robust_loss_param).ok().map(|l| {
                        Box::new(l)
                            as Box<dyn apex_solver::core::loss_functions::LossFunction + Send>
                    }),
                    "cauchy" => CauchyLoss::new(self.config.robust_loss_param)
                        .ok()
                        .map(|l| {
                            Box::new(l)
                                as Box<dyn apex_solver::core::loss_functions::LossFunction + Send>
                        }),
                    _ => None,
                };

            if self.config.optimize_intrinsics {
                // Use factor with intrinsics optimization
                let intr_var = format!("intr{}", obs.camera_idx);
                let factor = Box::new(ReprojectionFactorWithIntrinsics::new([obs.x, obs.y]));
                problem.add_residual_block(&[&cam_var, &pt_var, &intr_var], factor, loss);
            } else {
                // Use original factor with fixed intrinsics
                let cam = &self.cameras[obs.camera_idx];
                let intrinsics = cam.intrinsics_vector();
                let factor = Box::new(ReprojectionFactor::new([obs.x, obs.y], intrinsics));
                problem.add_residual_block(&[&cam_var, &pt_var], factor, loss);
            }
        }

        // Fix reference camera pose to anchor the coordinate system (gauge freedom)
        let ref_cam = self
            .config
            .reference_camera
            .min(self.cameras.len().saturating_sub(1));
        if !self.config.optimize_extrinsics {
            // Fix all cameras
            for i in 0..self.cameras.len() {
                for dof in 0..6 {
                    problem.fix_variable(&format!("cam{}", i), dof);
                }
            }
        } else {
            // Only fix reference camera
            for dof in 0..6 {
                problem.fix_variable(&format!("cam{}", ref_cam), dof);
            }
        }

        // Optionally fix all points
        if !self.config.optimize_points {
            for i in 0..self.points.len() {
                for dof in 0..3 {
                    problem.fix_variable(&format!("pt{}", i), dof);
                }
            }
        }

        // Configure solver
        let lm_config = LevenbergMarquardtConfig::new()
            .with_max_iterations(self.config.max_iterations)
            .with_cost_tolerance(self.config.cost_tolerance)
            .with_parameter_tolerance(self.config.parameter_tolerance)
            .with_gradient_tolerance(self.config.gradient_tolerance);

        // Run optimization with cost history observer
        let (cost_observer, cost_history) = CostHistoryObserver::new();
        let mut solver = LevenbergMarquardt::with_config(lm_config);
        solver.add_observer(cost_observer);
        let result = solver
            .optimize(&problem, &initial_values)
            .map_err(|e| JsValue::from_str(&format!("Optimization failed: {:?}", e)))?;

        // Extract optimized cameras (extrinsics and optionally intrinsics)
        let mut optimized_cameras = self.cameras.clone();
        for (i, cam) in optimized_cameras.iter_mut().enumerate() {
            // Extract extrinsics
            let var_name = format!("cam{}", i);
            if let Some(var) = result.parameters.get(&var_name) {
                let pose = var.to_vector();
                cam.translation = [pose[0], pose[1], pose[2]];
                cam.rotation = [pose[3], pose[4], pose[5], pose[6]];
            }
            // Extract intrinsics if optimized
            if self.config.optimize_intrinsics {
                let intr_name = format!("intr{}", i);
                if let Some(var) = result.parameters.get(&intr_name) {
                    let intr = var.to_vector();
                    cam.focal = [intr[0], intr[1]];
                    cam.principal = [intr[2], intr[3]];
                    cam.distortion = [intr[4], intr[5], intr[6], intr[7], intr[8]];
                }
            }
        }

        // Extract optimized points
        let mut optimized_points = self.points.clone();
        for (i, pt) in optimized_points.iter_mut().enumerate() {
            let var_name = format!("pt{}", i);
            if let Some(var) = result.parameters.get(&var_name) {
                let point = var.to_vector();
                *pt = [point[0], point[1], point[2]];
            }
        }

        // Check convergence status
        let converged = matches!(
            result.status,
            apex_solver::optimizer::OptimizationStatus::Converged
                | apex_solver::optimizer::OptimizationStatus::CostToleranceReached
                | apex_solver::optimizer::OptimizationStatus::ParameterToleranceReached
                | apex_solver::optimizer::OptimizationStatus::GradientToleranceReached
        );

        // Extract cost history from observer and prepend initial cost
        let mut costs = vec![result.initial_cost];
        if let Ok(guard) = cost_history.lock() {
            costs.extend(guard.iter().cloned());
        }

        // Build result
        let ba_result = BundleAdjustmentResult {
            cameras: optimized_cameras,
            points: optimized_points,
            initial_cost: result.initial_cost,
            final_cost: result.final_cost,
            iterations: result.iterations,
            converged,
            status: format!("{:?}", result.status),
            cost_history: costs,
            num_observations_used,
            num_observations_filtered,
            num_observations_filtered_by_frame,
        };

        // Serialize to JSON
        serde_json::to_string(&ba_result)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {}", e)))
    }

    /// Get the number of cameras.
    #[wasm_bindgen]
    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    /// Get the number of 3D points.
    #[wasm_bindgen]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Get the number of observations.
    #[wasm_bindgen]
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }
}

impl Default for WasmBundleAdjuster {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_params() {
        let cam = CameraParams {
            rotation: [1.0, 0.0, 0.0, 0.0], // identity quaternion
            translation: [0.0, 0.0, 0.0],
            focal: [500.0, 500.0],
            principal: [320.0, 240.0],
            distortion: [0.0, 0.0, 0.0, 0.0, 0.0],
        };

        let intrinsics = cam.intrinsics_vector();
        assert_eq!(intrinsics.len(), 9);
        assert_eq!(intrinsics[0], 500.0); // fx
        assert_eq!(intrinsics[2], 320.0); // cx
    }

    #[test]
    fn test_reprojection_factor() {
        let intrinsics = DVector::from_vec(vec![
            500.0, 500.0, // fx, fy
            320.0, 240.0, // cx, cy
            0.0, 0.0, 0.0, 0.0, 0.0, // no distortion
        ]);

        let factor = ReprojectionFactor::new([320.0, 240.0], intrinsics);

        // Camera at origin, identity rotation
        let pose = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        // Point at (0, 0, 1) in world
        let point = DVector::from_vec(vec![0.0, 0.0, 1.0]);

        let (residual, jacobian) = factor.linearize(&[pose, point], true);

        // Point at (0, 0, 1) should project to principal point (320, 240)
        assert!(residual[0].abs() < 1e-6);
        assert!(residual[1].abs() < 1e-6);
        assert!(jacobian.is_some());
    }

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.robust_loss, "huber");
    }

    #[test]
    fn test_wasm_bundle_adjuster_creation() {
        let ba = WasmBundleAdjuster::new();
        assert_eq!(ba.num_cameras(), 0);
        assert_eq!(ba.num_points(), 0);
        assert_eq!(ba.num_observations(), 0);
    }
}
