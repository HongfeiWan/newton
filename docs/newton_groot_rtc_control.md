# Newton GR00T RTC control

`tools/run_newton_groot_rtc_control.py` extends the dual Nero + Linker L10
scene in `debug/import_dual_nero_linker_l10.py`. It loads the local
`checkpoint-200000`, produces 32-step action chunks, and replans with the RTC
action-head path used by `probe_l10_rtc_trimmed_chunks.py`.

## Local assets

The runner uses these project-local paths by default:

- `checkpoints/groot/checkpoint-200000`
- `checkpoints/nvidia/Cosmos-Reason2-2B`
- `local_data/groot/smooth`
- `logs/groot_newton_rtc/trace.jsonl`

The Isaac-GR00T Python source is resolved from the Newton sibling directory
`../Isaac-GR00T` by default, or from `ISAAC_GROOT_ROOT` when that environment
variable is set.

The checkpoint, VLM, smooth dataset, and generated traces are ignored by Git.

Use the existing Newton conda environment. Isaac-GR00T pins the image
processor packages below; newer incompatible versions fail while constructing
the checkpoint processor.

```bash
conda activate newton
python -m pip install albumentations==1.4.18 albucore==0.0.17
```

Validate the copied assets, checkpoint modality contract, and one smooth frame
without loading Newton or the model:

```bash
python tools/run_newton_groot_rtc_control.py --validate-only --episode-index 8
```

## Live simulator images

This mode uses the current Newton D455 and D405 RGB buffers together with the
current simulated arm, hand, and end-effector state:

```bash
python tools/run_newton_groot_rtc_control.py \
  --viewer gl \
  --device cuda:0 \
  --policy-device cuda:0 \
  --image-source sim \
  --state-source sim \
  --start-policy
```

The policy receives the raw `640x480` D405 image. For the simulated D455, the
default `--sim-ego-roi` applies the scene camera's 2x ROI at `(0.50, 0.65)` so
the bottle, target rectangle, and arm match the framing of the training
`ego_view`. The resulting `640x400` RGB crop is sent directly to the processor;
the runner does not resize, pad, or letterbox it. Use `--no-sim-ego-roi` only
for camera-framing diagnostics. Smooth images are always passed through
unchanged, and the checkpoint processor performs its own resize and crop.

## Smooth episode images

Use recorded images while retaining the current Newton robot state:

```bash
python tools/run_newton_groot_rtc_control.py \
  --viewer gl \
  --image-source smooth \
  --state-source sim \
  --episode-index 8 \
  --smooth-frame-offset 0 \
  --start-policy
```

For a fully recorded observation, also set `--state-source smooth`. The image
and state sources are intentionally independent so recorded perception can be
tested against either recorded or simulated proprioception. An empty
`--instruction` uses the selected episode task; pass `--instruction TEXT` to
override it.

## RTC and execution

Defaults match the validated L10 deployment settings: 10 Hz actions, replan
every 8 executed actions, at most 24 overlap steps, 4 frozen steps, and an RTC
ramp rate of 3.0. Use `--no-rtc` for ordinary chunk replanning. Use
`--dry-run-policy` to test the Newton control loop without loading the model.

The default `--arm-control-mode eef_ik` treats decoded `eef_9d` as an absolute
TCP target in the checkpoint's rokae-base frame. At the first replan, the
runner aligns that frame to the current Newton `/right_revo2_flange` world
pose, maps each action target into Newton world coordinates, and applies it
through `NewtonLinkKinematicsModel` and the full-pose differential IK
controller. `arm_joint_target` is used only if EEF IK fails. Use
`--arm-control-mode joint_target` to select the old direct joint-alias path, or
`--no-arm-joint-fallback` to make an IK failure stop execution.

The Nero/L10 `rot6d` layout is the first two rotation-matrix columns in
column-major order: `[R00, R10, R20, R01, R11, R21]`. This matches the smooth
training data and the Harness deployment bridge; it is not the first-two-rows
layout used by some generic rot6d implementations.

This runner also selects the checkpoint/Harness right-arm initial pose instead
of the generic debug scene pose. The left arm continues to use the URDF initial
state. Pass the inherited `--initial-right-arm-q q1,...,q7` option only when
deliberately evaluating a different starting configuration.

The right L10 hand is initialized from the Harness checkpoint command pose so
its simulated reported state and wrist image begin inside the training
distribution. Override it with `--groot-initial-hand-q q1,...,q10` only for an
intentional state-distribution test.

For the Newton pinhole wrist camera, the runner uses a `72` degree vertical FOV
and a small connector-frame optical-axis correction. Together these reproduce
the D405's wider horizontal field and keep the nearby bottle in the lower-right
region seen in training. The generic scene's D405 body mount remains unchanged.

The default frame transform is fixed after the first observation. Use
`--eef-frame-update replan` only when deliberately recalibrating it at every
chunk. Per-tick arm and hand changes are bounded by `--max-arm-joint-step` and
`--max-hand-joint-step`. The trace records policy/world EEF targets, current
world TCP, IK status, position/orientation error, and the actual arm control
source for every executed action.

Policy execution is disabled until `--start-policy` is supplied. Every replan
and executed target is written to the JSONL trace unless `--no-policy-trace`
is set. For a bounded smoke test, add `--max-policy-steps 9`; step 8 performs
the first RTC replan with a previous action chunk.

Use `--dump-first-observation-dir PATH` to save the exact current `ego_view`
and `wrist_view` RGB arrays passed to the checkpoint processor. This is useful
for checking simulator/training camera alignment without changing image
preprocessing.

## Docker runtime

Build the GR00T inference layer on top of the existing direct-GPU image:

```bash
docker/build_groot_rtc.sh
```

On RTX 5090 hosts, `docker/run_groot_rtc.sh` automatically prefers the mounted
`conda_envs/newton` Python runtime because its CUDA 12.8 PyTorch build includes
`sm_120`. Set `NEWTON_GROOT_PYTHON` only when intentionally selecting another
Python runtime inside the container.

Run with live Newton images on GPU 0:

```bash
NEWTON_GROOT_GPU=0 docker/run_groot_rtc.sh \
  --viewer gl \
  --image-source sim \
  --state-source sim \
  --start-policy
```

For `--viewer gl`, run from the node3 desktop terminal so its current
`DISPLAY` and `.Xauthority` are available. The wrapper forwards both into the
container; do not hard-code a display number unless that X socket exists.

Run a recorded episode without opening a viewer:

```bash
NEWTON_GROOT_GPU=0 docker/run_groot_rtc.sh \
  --viewer null \
  --image-source smooth \
  --state-source smooth \
  --episode-index 8 \
  --start-policy
```
