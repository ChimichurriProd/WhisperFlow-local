# Per-model idle clips — generation brief

Goal: four 5 s idle-loop videos of Marvin, one per STT model, so the resting
look shows which model is loaded. The app looks for them as clips named
`idle_base` / `idle_small` / `idle_medium` / `idle_large`
(`_MODEL_IDLE_CLIP` in `app/pill.py`) — they activate automatically once the
frame folders exist under `assets/marvin/`.

**DONE 2026-07-11 with Kling 2.5** (slug `kling-25`, 5 s, 1:1, 720p,
140 credits/clip = 560 total). WAN 2.2 was the user's first choice (free in
the web app) but was broken server-side via the Magnific API (422, service
missing from their enum) — and note: unlimited does NOT apply via the API
anyway, every model draws credits there. The prompts below produced great
results with Kling 2.5 on the first try (no mouth hallucination, design held,
sparks + aura on the electric one).

Settings: duration **5 s**, aspect **1:1**. Start keyframe:
`assets/marvin/_design/expressions/exp_1_neutral.png` (on Magnific as
creation `VdnB6mBMMU`).

Processing: `scripts/build_model_idles.py <videos_dir>` — extracts frames via
ffmpeg, keys the black background (scipy border flood-fill), keeps aura/sparks
with smooth luminance alpha outside the head disc (super_saiyan-style), takes
every 2nd frame (61 per clip), resizes to 256×256 RGBA. No eyes.json needed
(idle clips don't drive the recording glow). Sync BOTH source and
`~/Library/WhisperFlow/assets/marvin/`, then restart the app.

## Prompts

### idle_base (sleepy)
Static locked-off camera, plain solid black background. A smooth matte grey
rounded robot head with small side ear nubs and tiny top knobs, two big solid
glowing green eyes with no pupils, heavy weary eyelids, and no mouth. The
robot is extremely sleepy and bored: its eyelids sag lower and lower until the
eyes are almost closed, the head slowly droops forward nodding off, then it
lazily catches itself and drowsily lifts back up, eyes reopening only halfway.
Very slow, heavy, drowsy movements. The dim green glow of the eyes fades
slightly as they close. The head stays perfectly centered and the same size in
frame, the camera never moves, the background stays pure black, nothing else
in frame.

### idle_small (relaxed)
Static locked-off camera, plain solid black background. A smooth matte grey
rounded robot head with small side ear nubs and tiny top knobs, two big solid
glowing green eyes with no pupils, relaxed half-lidded eyelids, and no mouth.
The robot is calm and relaxed: soft green eyes half open, one slow gentle
blink, the head sways very softly from side to side as if daydreaming
peacefully. Unhurried, smooth, tranquil movements. The head stays perfectly
centered and the same size in frame, the camera never moves, the background
stays pure black, nothing else in frame.

### idle_medium (awake)
Static locked-off camera, plain solid black background. A smooth matte grey
rounded robot head with small side ear nubs and tiny top knobs, two big solid
glowing green eyes with no pupils and no mouth. The robot is awake, alert and
attentive: eyes open bright green, glancing quickly left then right with
curious interest, small lively head tilts as it looks around, one quick blink.
Focused, perky, engaged energy. The head stays perfectly centered and the same
size in frame, the camera never moves, the background stays pure black,
nothing else in frame.

### idle_large (electric)
Static locked-off camera, plain solid black background. A smooth matte grey
rounded robot head with small side ear nubs and tiny top knobs, two big solid
glowing green eyes with no pupils and no mouth. The robot is supercharged with
electric energy: eyes wide open blazing intensely bright green, thin green
electric sparks and small lightning arcs crackle around the top and sides of
the head, a faint pulsing green energy aura glows around it. The head vibrates
subtly with excitement and makes quick energetic little tilts. High-voltage,
overclocked, powerful energy. The head stays perfectly centered and the same
size in frame, the camera never moves, the background stays pure black,
nothing else in frame.
