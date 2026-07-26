# SceneGenDeployBench-fr-iqa

Image-quality evaluators for SceneGenDeployBench. The shared image provides two separate runner catalogs:

- `fr_iqa` compares an aligned `data.image` and `output.image`.
- `3dgs_render_iqa` renders a Pano2Room/Graphdeco `output.3dgs` at the primary `data.camera_pose` and any available reference camera poses, then compares each view with its image. The primary pose becomes the 3DGS origin.

Both compute PSNR, SSIM, LPIPS, WS-PSNR, and DISTS by default. Higher is better for PSNR, WS-PSNR, and SSIM; lower is better for LPIPS and DISTS. WS-PSNR uses equirectangular spherical-area weighting. `3dgs_render_iqa` saves images using sample-based names such as `frame_000003.png`. Set `output_images: false` to calculate metrics without saving those images. References farther than `max_distance` from the primary camera are skipped; the default is 50 meters. When references are available, it also writes `metrics_by_distance.png`.

The image uses CUDA 12.8 and builds CUDA extensions for architectures 7.5, 8.0, 8.6, and 8.9, with PTX included for 8.9.

```bash
docker build -f runner_wrapper/Dockerfile -t scenegendeploybench-fr-iqa:local .
```
