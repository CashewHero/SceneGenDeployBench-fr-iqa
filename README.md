# SceneGenDeployBench-fr-iqa

Image-quality evaluators for SceneGenDeployBench. The shared image provides three separate runner catalogs:

- `fr_iqa` compares an aligned `data.image` and `output.image`.
- `3dgs_render_iqa` renders a Graphdeco-compatible `output.3dgs` at the primary `data.camera_pose` and any available reference camera poses, then compares each view with its image. The primary pose becomes the 3DGS origin.
- `3dgs_scale_calibration` searches for the generated scene scale that best aligns nearby rendered views with reference images and/or depth maps. It writes aggregate and per-reference CSV records for later analysis.

Scale calibration removes the highest-loss 5% of valid views before computing each objective mean. For each depth view it also excludes the 5% highest loss values.

The image evaluators compute PSNR, SSIM, LPIPS, WS-PSNR, and DISTS by default. Higher is better for PSNR, WS-PSNR, and SSIM; lower is better for LPIPS and DISTS. WS-PSNR uses equirectangular spherical-area weighting. `3dgs_render_iqa` saves images using sample-based names such as `frame_000003.png`. Set `output_images: false` to calculate metrics without saving those images. References are filtered by `max_distance`, sorted by distance, and capped by `max_references`. When multiple views are available, it also writes `metrics_by_distance.png`.

The image uses CUDA 12.8 and builds CUDA extensions for architectures 7.5, 8.0, 8.6, and 8.9, with PTX included for 8.9.

```bash
docker build -f runner_wrapper/Dockerfile -t scenegendeploybench-fr-iqa:local .
```
