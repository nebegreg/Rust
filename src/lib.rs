//! Ultimate Rotoscopy Core - High-Performance Rust Library
//!
//! This library provides performance-critical operations for the
//! Ultimate Rotoscopy application, including:
//!
//! - Edge detection and refinement
//! - Alpha channel processing
//! - Point cloud operations
//! - Image format conversions
//! - Multi-threaded batch processing

use pyo3::prelude::*;
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, IntoPyArray};
use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

/// Edge detection and refinement module
pub mod edge {
    use super::*;

    /// Sobel edge detection on a grayscale image
    #[pyfunction]
    pub fn sobel_edges<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        let img = image.as_array();
        let (h, w) = (img.shape()[0], img.shape()[1]);

        let mut edges = Array2::<f32>::zeros((h, w));

        // Sobel kernels
        let gx: [[f32; 3]; 3] = [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ];

        let gy: [[f32; 3]; 3] = [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ];

        // Parallel edge detection
        edges
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                if y > 0 && y < h - 1 {
                    for x in 1..w - 1 {
                        let mut sum_x = 0.0f32;
                        let mut sum_y = 0.0f32;

                        for ky in 0..3 {
                            for kx in 0..3 {
                                let pixel = img[[y + ky - 1, x + kx - 1]];
                                sum_x += pixel * gx[ky][kx];
                                sum_y += pixel * gy[ky][kx];
                            }
                        }

                        row[x] = (sum_x * sum_x + sum_y * sum_y).sqrt();
                    }
                }
            });

        edges.into_pyarray(py)
    }

    /// Canny edge detection
    #[pyfunction]
    pub fn canny_edges<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<f32>,
        low_threshold: f32,
        high_threshold: f32,
    ) -> &'py PyArray2<f32> {
        let img = image.as_array();
        let (_h, _w) = (img.shape()[0], img.shape()[1]);

        // Step 1: Gaussian blur
        let blurred = gaussian_blur(&img);

        // Step 2: Gradient calculation
        let (grad_mag, grad_dir) = compute_gradients(&blurred);

        // Step 3: Non-maximum suppression
        let suppressed = non_max_suppression(&grad_mag, &grad_dir);

        // Step 4: Double threshold and edge tracking
        let edges = hysteresis_threshold(&suppressed, low_threshold, high_threshold);

        edges.into_pyarray(py)
    }

    fn gaussian_blur(img: &ndarray::ArrayView2<f32>) -> Array2<f32> {
        let (h, w) = (img.shape()[0], img.shape()[1]);
        let mut result = Array2::<f32>::zeros((h, w));

        let kernel: [[f32; 5]; 5] = [
            [1.0, 4.0, 6.0, 4.0, 1.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ];
        let kernel_sum: f32 = 256.0;

        for y in 2..h - 2 {
            for x in 2..w - 2 {
                let mut sum = 0.0f32;
                for ky in 0..5 {
                    for kx in 0..5 {
                        sum += img[[y + ky - 2, x + kx - 2]] * kernel[ky][kx];
                    }
                }
                result[[y, x]] = sum / kernel_sum;
            }
        }

        result
    }

    fn compute_gradients(img: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let (h, w) = (img.shape()[0], img.shape()[1]);
        let mut mag = Array2::<f32>::zeros((h, w));
        let mut dir = Array2::<f32>::zeros((h, w));

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let gx = img[[y, x + 1]] - img[[y, x - 1]];
                let gy = img[[y + 1, x]] - img[[y - 1, x]];

                mag[[y, x]] = (gx * gx + gy * gy).sqrt();
                dir[[y, x]] = gy.atan2(gx);
            }
        }

        (mag, dir)
    }

    fn non_max_suppression(mag: &Array2<f32>, dir: &Array2<f32>) -> Array2<f32> {
        let (h, w) = (mag.shape()[0], mag.shape()[1]);
        let mut result = Array2::<f32>::zeros((h, w));

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let angle = dir[[y, x]].to_degrees();
                let angle = if angle < 0.0 { angle + 180.0 } else { angle };

                let (n1, n2) = if angle < 22.5 || angle >= 157.5 {
                    (mag[[y, x - 1]], mag[[y, x + 1]])
                } else if angle < 67.5 {
                    (mag[[y - 1, x + 1]], mag[[y + 1, x - 1]])
                } else if angle < 112.5 {
                    (mag[[y - 1, x]], mag[[y + 1, x]])
                } else {
                    (mag[[y - 1, x - 1]], mag[[y + 1, x + 1]])
                };

                if mag[[y, x]] >= n1 && mag[[y, x]] >= n2 {
                    result[[y, x]] = mag[[y, x]];
                }
            }
        }

        result
    }

    fn hysteresis_threshold(img: &Array2<f32>, low: f32, high: f32) -> Array2<f32> {
        let (h, w) = (img.shape()[0], img.shape()[1]);
        let mut result = Array2::<f32>::zeros((h, w));

        // Mark strong and weak edges
        for y in 0..h {
            for x in 0..w {
                if img[[y, x]] >= high {
                    result[[y, x]] = 1.0;
                } else if img[[y, x]] >= low {
                    result[[y, x]] = 0.5;
                }
            }
        }

        // Connect weak edges to strong edges
        let mut changed = true;
        while changed {
            changed = false;
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    if result[[y, x]] == 0.5 {
                        // Check 8-neighbors for strong edge
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let ny = (y as i32 + dy) as usize;
                                let nx = (x as i32 + dx) as usize;
                                if result[[ny, nx]] == 1.0 {
                                    result[[y, x]] = 1.0;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Remove weak edges not connected to strong
        for y in 0..h {
            for x in 0..w {
                if result[[y, x]] == 0.5 {
                    result[[y, x]] = 0.0;
                }
            }
        }

        result
    }

    /// Refine edges using guided filtering
    #[pyfunction]
    pub fn refine_edges<'py>(
        py: Python<'py>,
        mask: PyReadonlyArray2<f32>,
        guide: PyReadonlyArray3<f32>,
        radius: usize,
        epsilon: f32,
    ) -> &'py PyArray2<f32> {
        let mask_arr = mask.as_array();
        let guide_arr = guide.as_array();
        let (h, w) = (mask_arr.shape()[0], mask_arr.shape()[1]);

        // Simplified guided filter implementation
        let mut result = Array2::<f32>::zeros((h, w));

        let r = radius as i32;

        for y in 0..h {
            for x in 0..w {
                let mut sum_weight = 0.0f32;
                let mut sum_value = 0.0f32;

                let y0 = (y as i32 - r).max(0) as usize;
                let y1 = (y as i32 + r + 1).min(h as i32) as usize;
                let x0 = (x as i32 - r).max(0) as usize;
                let x1 = (x as i32 + r + 1).min(w as i32) as usize;

                for ny in y0..y1 {
                    for nx in x0..x1 {
                        // Calculate weight based on guide image similarity
                        let mut color_dist = 0.0f32;
                        for c in 0..3 {
                            let d = guide_arr[[y, x, c]] - guide_arr[[ny, nx, c]];
                            color_dist += d * d;
                        }

                        let weight = (-color_dist / (2.0 * epsilon * epsilon)).exp();
                        sum_weight += weight;
                        sum_value += weight * mask_arr[[ny, nx]];
                    }
                }

                result[[y, x]] = if sum_weight > 0.0 {
                    sum_value / sum_weight
                } else {
                    mask_arr[[y, x]]
                };
            }
        }

        result.into_pyarray(py)
    }
}

/// Alpha channel processing module
pub mod alpha {
    use super::*;

    /// Morphological operations on alpha channel
    #[pyfunction]
    pub fn morphological_cleanup<'py>(
        py: Python<'py>,
        alpha: PyReadonlyArray2<f32>,
        kernel_size: usize,
        operation: &str,
    ) -> &'py PyArray2<f32> {
        let img = alpha.as_array();
        let (h, w) = (img.shape()[0], img.shape()[1]);

        let result = match operation {
            "erode" => morphological_erode(&img, kernel_size),
            "dilate" => morphological_dilate(&img, kernel_size),
            "open" => {
                let eroded = morphological_erode(&img, kernel_size);
                morphological_dilate(&eroded.view(), kernel_size)
            }
            "close" => {
                let dilated = morphological_dilate(&img, kernel_size);
                morphological_erode(&dilated.view(), kernel_size)
            }
            _ => Array2::from_shape_fn((h, w), |(y, x)| img[[y, x]]),
        };

        result.into_pyarray(py)
    }

    fn morphological_erode(img: &ndarray::ArrayView2<f32>, kernel_size: usize) -> Array2<f32> {
        let (h, w) = (img.shape()[0], img.shape()[1]);
        let mut result = Array2::<f32>::zeros((h, w));
        let k = kernel_size / 2;

        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..w {
                    let mut min_val = f32::MAX;

                    let y0 = y.saturating_sub(k);
                    let y1 = (y + k + 1).min(h);
                    let x0 = x.saturating_sub(k);
                    let x1 = (x + k + 1).min(w);

                    for ny in y0..y1 {
                        for nx in x0..x1 {
                            min_val = min_val.min(img[[ny, nx]]);
                        }
                    }

                    row[x] = min_val;
                }
            });

        result
    }

    fn morphological_dilate(img: &ndarray::ArrayView2<f32>, kernel_size: usize) -> Array2<f32> {
        let (h, w) = (img.shape()[0], img.shape()[1]);
        let mut result = Array2::<f32>::zeros((h, w));
        let k = kernel_size / 2;

        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..w {
                    let mut max_val = f32::MIN;

                    let y0 = y.saturating_sub(k);
                    let y1 = (y + k + 1).min(h);
                    let x0 = x.saturating_sub(k);
                    let x1 = (x + k + 1).min(w);

                    for ny in y0..y1 {
                        for nx in x0..x1 {
                            max_val = max_val.max(img[[ny, nx]]);
                        }
                    }

                    row[x] = max_val;
                }
            });

        result
    }

    /// Feather alpha edges
    #[pyfunction]
    pub fn feather_alpha<'py>(
        py: Python<'py>,
        alpha: PyReadonlyArray2<f32>,
        radius: f32,
    ) -> &'py PyArray2<f32> {
        let img = alpha.as_array();
        let (h, w) = (img.shape()[0], img.shape()[1]);

        let r = radius.ceil() as i32;
        let mut result = Array2::<f32>::zeros((h, w));

        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..w {
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    let y0 = (y as i32 - r).max(0) as usize;
                    let y1 = (y as i32 + r + 1).min(h as i32) as usize;
                    let x0 = (x as i32 - r).max(0) as usize;
                    let x1 = (x as i32 + r + 1).min(w as i32) as usize;

                    for ny in y0..y1 {
                        for nx in x0..x1 {
                            let dx = x as f32 - nx as f32;
                            let dy = y as f32 - ny as f32;
                            let dist = (dx * dx + dy * dy).sqrt();

                            if dist <= radius {
                                let weight = 1.0 - dist / radius;
                                sum += img[[ny, nx]] * weight;
                                weight_sum += weight;
                            }
                        }
                    }

                    row[x] = if weight_sum > 0.0 {
                        sum / weight_sum
                    } else {
                        img[[y, x]]
                    };
                }
            });

        result.into_pyarray(py)
    }

    /// Apply premultiplication to RGBA image
    #[pyfunction]
    pub fn premultiply<'py>(
        py: Python<'py>,
        rgb: PyReadonlyArray3<f32>,
        alpha: PyReadonlyArray2<f32>,
    ) -> &'py PyArray3<f32> {
        let rgb_arr = rgb.as_array();
        let alpha_arr = alpha.as_array();
        let (h, w) = (rgb_arr.shape()[0], rgb_arr.shape()[1]);

        let mut result = Array3::<f32>::zeros((h, w, 4));

        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..w {
                    let a = alpha_arr[[y, x]];
                    row[[x, 0]] = rgb_arr[[y, x, 0]] * a;
                    row[[x, 1]] = rgb_arr[[y, x, 1]] * a;
                    row[[x, 2]] = rgb_arr[[y, x, 2]] * a;
                    row[[x, 3]] = a;
                }
            });

        result.into_pyarray(py)
    }

    /// Remove premultiplication from RGBA image
    #[pyfunction]
    pub fn unpremultiply<'py>(
        py: Python<'py>,
        rgba: PyReadonlyArray3<f32>,
    ) -> &'py PyArray3<f32> {
        let img = rgba.as_array();
        let (h, w) = (img.shape()[0], img.shape()[1]);

        let mut result = Array3::<f32>::zeros((h, w, 3));

        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..w {
                    let a = img[[y, x, 3]];
                    if a > 1e-6 {
                        row[[x, 0]] = img[[y, x, 0]] / a;
                        row[[x, 1]] = img[[y, x, 1]] / a;
                        row[[x, 2]] = img[[y, x, 2]] / a;
                    } else {
                        row[[x, 0]] = 0.0;
                        row[[x, 1]] = 0.0;
                        row[[x, 2]] = 0.0;
                    }
                }
            });

        result.into_pyarray(py)
    }
}

/// Depth processing module
pub mod depth {
    use super::*;

    /// Compute normals from depth map
    #[pyfunction]
    pub fn depth_to_normals<'py>(
        py: Python<'py>,
        depth: PyReadonlyArray2<f32>,
    ) -> &'py PyArray3<f32> {
        let depth_arr = depth.as_array();
        let (h, w) = (depth_arr.shape()[0], depth_arr.shape()[1]);

        let mut normals = Array3::<f32>::zeros((h, w, 3));

        normals
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..w {
                    if y > 0 && y < h - 1 && x > 0 && x < w - 1 {
                        let dzdx = depth_arr[[y, x + 1]] - depth_arr[[y, x - 1]];
                        let dzdy = depth_arr[[y + 1, x]] - depth_arr[[y - 1, x]];

                        let nx = -dzdx;
                        let ny = -dzdy;
                        let nz = 1.0f32;

                        let len = (nx * nx + ny * ny + nz * nz).sqrt();

                        row[[x, 0]] = nx / len;
                        row[[x, 1]] = ny / len;
                        row[[x, 2]] = nz / len;
                    } else {
                        row[[x, 0]] = 0.0;
                        row[[x, 1]] = 0.0;
                        row[[x, 2]] = 1.0;
                    }
                }
            });

        normals.into_pyarray(py)
    }

    /// Bilateral filter for edge-aware depth smoothing
    #[pyfunction]
    pub fn bilateral_filter<'py>(
        py: Python<'py>,
        depth: PyReadonlyArray2<f32>,
        guide: PyReadonlyArray3<f32>,
        spatial_sigma: f32,
        range_sigma: f32,
        radius: i32,
    ) -> &'py PyArray2<f32> {
        let depth_arr = depth.as_array();
        let guide_arr = guide.as_array();
        let (h, w) = (depth_arr.shape()[0], depth_arr.shape()[1]);

        let mut result = Array2::<f32>::zeros((h, w));

        let spatial_coeff = -1.0 / (2.0 * spatial_sigma * spatial_sigma);
        let range_coeff = -1.0 / (2.0 * range_sigma * range_sigma);

        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..w {
                    let mut sum_weight = 0.0f32;
                    let mut sum_value = 0.0f32;

                    let y0 = (y as i32 - radius).max(0) as usize;
                    let y1 = (y as i32 + radius + 1).min(h as i32) as usize;
                    let x0 = (x as i32 - radius).max(0) as usize;
                    let x1 = (x as i32 + radius + 1).min(w as i32) as usize;

                    for ny in y0..y1 {
                        for nx in x0..x1 {
                            // Spatial weight
                            let dx = x as f32 - nx as f32;
                            let dy = y as f32 - ny as f32;
                            let spatial_weight = (spatial_coeff * (dx * dx + dy * dy)).exp();

                            // Range weight (based on guide image)
                            let mut range_dist = 0.0f32;
                            for c in 0..3 {
                                let d = guide_arr[[y, x, c]] - guide_arr[[ny, nx, c]];
                                range_dist += d * d;
                            }
                            let range_weight = (range_coeff * range_dist).exp();

                            let weight = spatial_weight * range_weight;
                            sum_weight += weight;
                            sum_value += weight * depth_arr[[ny, nx]];
                        }
                    }

                    row[x] = if sum_weight > 0.0 {
                        sum_value / sum_weight
                    } else {
                        depth_arr[[y, x]]
                    };
                }
            });

        result.into_pyarray(py)
    }
}

/// Point cloud operations module
pub mod pointcloud {
    use super::*;

    /// Back-project depth map to 3D points
    #[pyfunction]
    pub fn depth_to_pointcloud<'py>(
        py: Python<'py>,
        depth: PyReadonlyArray2<f32>,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
    ) -> &'py PyArray2<f32> {
        let depth_arr = depth.as_array();
        let (h, w) = (depth_arr.shape()[0], depth_arr.shape()[1]);

        // Count valid points
        let mut valid_count = 0usize;
        for y in 0..h {
            for x in 0..w {
                if depth_arr[[y, x]] > 0.0 {
                    valid_count += 1;
                }
            }
        }

        let mut points = Array2::<f32>::zeros((valid_count, 3));
        let mut idx = 0;

        for y in 0..h {
            for x in 0..w {
                let z = depth_arr[[y, x]];
                if z > 0.0 {
                    let px = (x as f32 - cx) * z / fx;
                    let py = (y as f32 - cy) * z / fy;

                    points[[idx, 0]] = px;
                    points[[idx, 1]] = py;
                    points[[idx, 2]] = z;
                    idx += 1;
                }
            }
        }

        points.into_pyarray(py)
    }

    /// Compute point cloud normals using PCA
    #[pyfunction]
    pub fn compute_pointcloud_normals<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<f32>,
        k_neighbors: usize,
    ) -> &'py PyArray2<f32> {
        let pts = points.as_array();
        let n_points = pts.shape()[0];

        let mut normals = Array2::<f32>::zeros((n_points, 3));

        // Simple normal estimation (in production, use KD-tree for efficiency)
        normals
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut normal)| {
                // Find k nearest neighbors (brute force for simplicity)
                let mut distances: Vec<(usize, f32)> = (0..n_points)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let dx = pts[[i, 0]] - pts[[j, 0]];
                        let dy = pts[[i, 1]] - pts[[j, 1]];
                        let dz = pts[[i, 2]] - pts[[j, 2]];
                        (j, dx * dx + dy * dy + dz * dz)
                    })
                    .collect();

                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let k = k_neighbors.min(distances.len());
                if k < 3 {
                    normal[0] = 0.0;
                    normal[1] = 0.0;
                    normal[2] = 1.0;
                    return;
                }

                // Compute covariance matrix
                let mut centroid = [0.0f32; 3];
                for j in 0..k {
                    let idx = distances[j].0;
                    centroid[0] += pts[[idx, 0]];
                    centroid[1] += pts[[idx, 1]];
                    centroid[2] += pts[[idx, 2]];
                }
                centroid[0] /= k as f32;
                centroid[1] /= k as f32;
                centroid[2] /= k as f32;

                let mut cov = [[0.0f32; 3]; 3];
                for j in 0..k {
                    let idx = distances[j].0;
                    let dx = pts[[idx, 0]] - centroid[0];
                    let dy = pts[[idx, 1]] - centroid[1];
                    let dz = pts[[idx, 2]] - centroid[2];

                    cov[0][0] += dx * dx;
                    cov[0][1] += dx * dy;
                    cov[0][2] += dx * dz;
                    cov[1][1] += dy * dy;
                    cov[1][2] += dy * dz;
                    cov[2][2] += dz * dz;
                }

                cov[1][0] = cov[0][1];
                cov[2][0] = cov[0][2];
                cov[2][1] = cov[1][2];

                // Simple power iteration for smallest eigenvector
                let mut v = [1.0f32, 0.0, 0.0];
                for _ in 0..10 {
                    let mut new_v = [
                        cov[0][0] * v[0] + cov[0][1] * v[1] + cov[0][2] * v[2],
                        cov[1][0] * v[0] + cov[1][1] * v[1] + cov[1][2] * v[2],
                        cov[2][0] * v[0] + cov[2][1] * v[1] + cov[2][2] * v[2],
                    ];

                    let len = (new_v[0] * new_v[0] + new_v[1] * new_v[1] + new_v[2] * new_v[2]).sqrt();
                    if len > 1e-6 {
                        new_v[0] /= len;
                        new_v[1] /= len;
                        new_v[2] /= len;
                    }
                    v = new_v;
                }

                // Orient normal towards camera (positive Z)
                if v[2] < 0.0 {
                    v[0] = -v[0];
                    v[1] = -v[1];
                    v[2] = -v[2];
                }

                normal[0] = v[0];
                normal[1] = v[1];
                normal[2] = v[2];
            });

        normals.into_pyarray(py)
    }
}

/// EXR utilities module
pub mod exr {
    use super::*;

    /// Convert image to half-float format for EXR
    #[pyfunction]
    pub fn to_half_float<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<u16> {
        let img = image.as_array();
        let (h, w) = (img.shape()[0], img.shape()[1]);

        let mut result = ndarray::Array2::<u16>::zeros((h, w));

        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..w {
                    row[x] = f32_to_f16(img[[y, x]]);
                }
            });

        result.into_pyarray(py)
    }

    fn f32_to_f16(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x7FFFFF;

        if exp == 0 {
            // Subnormal or zero
            0
        } else if exp == 0xFF {
            // Inf or NaN
            ((sign << 15) | 0x7C00 | (mantissa >> 13)) as u16
        } else {
            let new_exp = exp - 127 + 15;
            if new_exp >= 31 {
                // Overflow -> Inf
                ((sign << 15) | 0x7C00) as u16
            } else if new_exp <= 0 {
                // Underflow -> 0
                0
            } else {
                ((sign << 15) | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
            }
        }
    }
}

/// Python module initialization
#[pymodule]
fn rotoscopy_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Edge detection
    m.add_function(wrap_pyfunction!(edge::sobel_edges, m)?)?;
    m.add_function(wrap_pyfunction!(edge::canny_edges, m)?)?;
    m.add_function(wrap_pyfunction!(edge::refine_edges, m)?)?;

    // Alpha processing
    m.add_function(wrap_pyfunction!(alpha::morphological_cleanup, m)?)?;
    m.add_function(wrap_pyfunction!(alpha::feather_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(alpha::premultiply, m)?)?;
    m.add_function(wrap_pyfunction!(alpha::unpremultiply, m)?)?;

    // Depth processing
    m.add_function(wrap_pyfunction!(depth::depth_to_normals, m)?)?;
    m.add_function(wrap_pyfunction!(depth::bilateral_filter, m)?)?;

    // Point cloud
    m.add_function(wrap_pyfunction!(pointcloud::depth_to_pointcloud, m)?)?;
    m.add_function(wrap_pyfunction!(pointcloud::compute_pointcloud_normals, m)?)?;

    // EXR utilities
    m.add_function(wrap_pyfunction!(exr::to_half_float, m)?)?;

    Ok(())
}
