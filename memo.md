メッシュだけ表示する

python scripts/render_slice_figure.py \
  --mesh meshes/bunny_zipper_wataertight.ply \
  --mode mesh \
  --plane-axis z \
  --plane-value 0.12 \
  --plane-size 2.0 \
  --output figures/bunny_mesh_only.png \
  --show-cut-outline

  2. 断面画像を貼る
  python scripts/render_slice_figure.py \
  --mesh meshes/processed/bunny.obj \
  --mode slice \
  --slice-png figures/slice_gt.png \
  --plane-axis z \
  --plane-value 0.12 \
  --plane-size 2.0 \
  --output figures/bunny_gt_slice.png \
  --show-cut-outline \
  --colorbar-output figures/colorbar_gt.png \
  --colorbar-meta figures/colorbar_gt.json \
  --colorbar-cmap coolwarm \
  --colorbar-vmin -1.0 \
  --colorbar-vmax 1.0

  Neural Cache / NC-WoS の slice をメッシュに貼る：

  python scripts/render_slice_figure.py \
  --json figures/zebra_k8_m2.json \
  --mesh procedural_bumpy_sphere \
  --value-column nc_wos_mean \
  --texture-output figures/zebra_k8_nc_texture.png \
  --texture-meta figures/zebra_k8_nc_texture.json \
  --render-output figures/zebra_k8_nc_render.png \
  --colorbar-output figures/zebra_k8_nc_colorbar.png \
  --colorbar-meta figures/zebra_k8_nc_colorbar.json \
  --cmap coolwarm \
  --show-cut

  signed error
  python scripts/render_slice_figure.py \
  --json figures/zebra_k8_m2.json \
  --mesh procedural_bumpy_sphere \
  --value-column nc_wos_mean \
  --subtract-column analytic_value \
  --symmetric-range \
  --texture-output figures/zebra_k8_nc_error_texture.png \
  --texture-meta figures/zebra_k8_nc_error_texture.json \
  --render-output figures/zebra_k8_nc_error_render.png \
  --colorbar-output figures/zebra_k8_error_colorbar.png \
  --colorbar-meta figures/zebra_k8_error_colorbar.json \
  --cmap coolwarm \
  --show-cut

  mesh only：

  python scripts/render_slice_figure.py \
  --json figures/zebra_k8_m2.json \
  --mesh procedural_bumpy_sphere \
  --mesh-only \
  --render-output figures/mesh_only.png \
  --show-cut

GT field:

python scripts/render_slice_figure.py \
  --json results/bunny_charges_medium_z012_medium_m4_c64_r48.json \
  --mesh meshes/bunny_zipper_wataertight.ply \
  --value-column analytic_value \
  --texture-output figures/bunny_charges_shell_medium_k8_gt_texture.png \
  --texture-meta figures/bunny_charges_shell_medium_k8_gt_texture.json \
  --render-output figures/bunny_charges_shell_medium_k8_gt_render.png \
  --colorbar-output figures/bunny_charges_shell_medium_k8_gt_colorbar.png \
  --colorbar-meta figures/bunny_charges_shell_medium_k8_gt_colorbar.json \
  --cmap coolwarm \
  --show-cut

  python scripts/render_slice_figure.py \
  --json results/bunny_charges_medium_z012_medium_m4_c64_r48.json \
  --mesh meshes/bunny_zipper_wataertight.ply \
  --value-column nc_wos_mean \
  --subtract-column analytic_value \
  --symmetric-range \
  --texture-output figures/bunny_charges_shell_k8_nc_error_texture.png \
  --texture-meta figures/bunny_charges_shell_k8_nc_error_texture.json \
  --render-output figures/bunny_charges_shell_k8_nc_error_render.png \
  --colorbar-output figures/bunny_charges_shell_k8_error_colorbar.png \
  --colorbar-meta figures/bunny_charges_shell_k8_error_colorbar.json \
  --cmap coolwarm \
  --show-cut