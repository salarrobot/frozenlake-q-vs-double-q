"""Record a trained agent playing FrozenLake as an MP4/GIF video.

Two renderers are supported:

* ``"original"`` — Gymnasium's own ``rgb_array`` renderer with the FrozenLake
  sprite artwork (the elf, ice tiles, cracked-ice holes and the gift-box goal).
  Requires ``pygame``; rendered fully off-screen via the SDL ``dummy`` driver so
  no window pops up.
* ``"simple"`` — a self-contained Pillow grid (coloured tiles + a disc marker),
  used automatically as a fallback when ``pygame`` is unavailable.

Either way each frame gets a caption bar (episode, step, state, chosen action,
running reward and the final outcome).  Video is written with ``imageio`` — MP4
via the bundled ``imageio-ffmpeg`` binary, with an automatic GIF fallback.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Keep pygame quiet and head-less friendly before it is ever imported.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:                       # Pillow is required for the caption overlay.
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except Exception:          # pragma: no cover
    _HAVE_PIL = False

from .agents.base import TabularAgent
from .config import ExperimentConfig
from .environment import ACTIONS, MAPS, make_env, state_to_rowcol
from .utils import ensure_dir


# --------------------------------------------------------------------------- #
# Caption overlay
# --------------------------------------------------------------------------- #
def _font(size: int):
    for name in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# Tile colours for the self-contained grid renderer (no pygame dependency).
_TILE_COLORS = {
    "F": (160, 208, 232),   # frozen ice
    "S": (88, 168, 110),    # start
    "G": (236, 196, 84),    # goal
    "H": (26, 56, 96),      # hole (water)
}


def _centered_text(draw, center, text, font, fill) -> None:
    try:
        draw.text(center, text, font=font, fill=fill, anchor="mm")
    except Exception:        # bitmap fallback font has no anchor support
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((center[0] - w / 2, center[1] - h / 2), text, font=font, fill=fill)


def _render_grid(map_rows, state: int, ncols: int, cell: int) -> np.ndarray:
    """Draw the FrozenLake board with the agent marker, returning an RGB array."""
    width, height = ncols * cell, len(map_rows) * cell
    img = Image.new("RGB", (width, height), (10, 20, 40))
    draw = ImageDraw.Draw(img)
    font = _font(int(cell * 0.34))
    for r, row in enumerate(map_rows):
        for c, ch in enumerate(row):
            x0, y0 = c * cell, r * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell],
                           fill=_TILE_COLORS.get(ch, _TILE_COLORS["F"]),
                           outline=(255, 255, 255), width=2)
            if ch in ("S", "G", "H"):
                col = (235, 235, 235) if ch == "H" else (35, 35, 35)
                _centered_text(draw, (x0 + cell / 2, y0 + cell / 2), ch, font, col)
    # Agent marker.
    ar, ac = divmod(state, ncols)
    cx, cy = ac * cell + cell / 2, ar * cell + cell / 2
    rad = cell * 0.30
    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                 fill=(214, 64, 64), outline=(255, 255, 255), width=3)
    return np.asarray(img)


def _compose(frame: np.ndarray, scale: int, caption: str, sub: str,
             accent=(120, 200, 255), smooth: bool = False) -> np.ndarray:
    """Upscale a frame and add a two-line caption bar; return an even-sized RGB array."""
    if not _HAVE_PIL:
        return frame
    img = Image.fromarray(frame).convert("RGB")
    if scale != 1:
        resample = Image.BILINEAR if smooth else Image.NEAREST
        img = img.resize((img.width * scale, img.height * scale), resample)
    w, h = img.size

    fs = max(15, w // 26)
    font, sfont = _font(fs), _font(int(fs * 0.82))
    bar = int(fs * 2.7)
    bar += bar % 2                       # keep total height even (MP4-friendly)

    canvas = Image.new("RGB", (w, h + bar), (17, 18, 26))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(0, h), (w, h)], fill=accent, width=2)
    draw.text((10, h + 5), caption, fill=(255, 255, 255), font=font)
    if sub:
        draw.text((10, h + 7 + fs), sub, fill=accent, font=sfont)

    arr = np.asarray(canvas)
    if arr.shape[1] % 2:                 # guarantee even width too
        arr = arr[:, :-1, :]
    return arr


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def record_agent_video(agent: TabularAgent, config: ExperimentConfig, out_path: str,
                       n_episodes: int = 3, fps: int = 3, scale: int | None = None,
                       hold_seconds: float = 1.3, seed: int | None = None,
                       video_format: str = "mp4", graphics: str = "auto") -> Path:
    """Render ``n_episodes`` of the greedy policy to a video file.

    ``graphics`` selects the renderer: ``"original"`` (Gymnasium sprite art via
    pygame), ``"simple"`` (the built-in Pillow grid) or ``"auto"`` (original with
    a silent fallback to simple).  Returns the path actually written (the
    extension may switch to ``.gif`` if MP4 encoding is unavailable).
    """
    ncols = 4 if config.map_name == "4x4" else 8
    cell = 120 if config.map_name == "4x4" else 60   # simple-renderer tile size
    map_rows = MAPS[config.map_name]
    name = type(agent).name
    seed = config.seed + 99_000 if seed is None else seed

    use_original = graphics in ("auto", "original")
    prev_driver = os.environ.get("SDL_VIDEODRIVER")
    if use_original:
        # Render fully off-screen so no window appears during recording.
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    def _new_env(render_mode):
        e = make_env(config.map_name, config.is_slippery, render_mode=render_mode)
        e.reset(seed=seed)
        e.action_space.seed(seed)
        return e

    env = _new_env("rgb_array" if use_original else None)
    if use_original:                       # probe the sprite renderer
        try:
            env.render()
        except Exception as exc:
            if graphics == "original":
                env.close()
                _restore_driver(prev_driver)
                raise
            print(f"  [info] original graphics unavailable "
                  f"({exc.__class__.__name__}); falling back to simple renderer.")
            use_original = False
            env.close()
            _restore_driver(prev_driver)
            env = _new_env(None)

    if scale is None:
        # Bring the board to ~512 px wide for legible captions.
        scale = (2 if config.map_name == "4x4" else 1) if use_original else 1
    smooth = use_original                   # bilinear upscaling for sprite art

    def board(s):
        # env's internal state always equals `s` at every call site below.
        return env.render() if use_original else _render_grid(map_rows, s, ncols, cell)

    hold = max(1, int(round(hold_seconds * fps)))
    intro_hold = max(1, int(round(0.6 * fps)))
    frames: list[np.ndarray] = []
    successes = 0

    try:
        for ep in range(n_episodes):
            state, _ = env.reset()
            terminated = truncated = False
            total = 0.0
            step = 0
            head = f"{name}  |  FrozenLake {config.map_name}  slippery={config.is_slippery}"

            # Intro: agent on the start tile.
            frames.extend([_compose(
                board(state), scale,
                f"Episode {ep + 1}/{n_episodes}   —   START",
                f"{head}   |   start at state {state}", smooth=smooth)] * intro_hold)

            while not (terminated or truncated):
                action = agent.greedy_action(state)
                row, col = state_to_rowcol(state, ncols)
                # Decision frame: marker sits on `state`, about to take `action`.
                # The result (and any slip) appears on the next frame.
                frames.append(_compose(
                    board(state), scale,
                    f"Episode {ep + 1}/{n_episodes}   step {step + 1}   reward {total:.0f}",
                    f"at state {row*ncols+col} (r{row},c{col})   ->   "
                    f"action {action} = {ACTIONS[action]}", smooth=smooth))
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                step += 1

            success = total > 0
            successes += int(success)
            outcome = ("REACHED GOAL" if success else
                       ("TIMED OUT" if truncated else "FELL IN HOLE"))
            # Freeze the final frame so the outcome is readable.
            final = _compose(
                board(state), scale,
                f"Episode {ep + 1}/{n_episodes}   {outcome}   ({step} steps)",
                f"{head}   |   total reward {total:.0f}",
                accent=(120, 230, 140) if success else (240, 120, 120), smooth=smooth)
            frames.extend([final] * hold)
    finally:
        env.close()
        _restore_driver(prev_driver)

    out = _write_video(frames, out_path, fps, video_format)
    renderer = "original sprites" if use_original else "simple grid"
    print(f"  recorded {n_episodes} episodes ({successes} solved, {renderer}) -> {out}")
    return out


def _restore_driver(prev: str | None) -> None:
    """Restore the SDL video driver env var after off-screen recording."""
    if prev is None:
        os.environ.pop("SDL_VIDEODRIVER", None)
    else:
        os.environ["SDL_VIDEODRIVER"] = prev


def _write_video(frames: list[np.ndarray], out_path: str, fps: int,
                 video_format: str) -> Path:
    """Write frames to MP4 (preferred) or GIF, returning the path written."""
    import imageio.v2 as imageio

    out = Path(out_path)
    ensure_dir(out.parent)

    if video_format == "gif":
        gif = out.with_suffix(".gif")
        imageio.mimsave(gif, frames, fps=fps, loop=0)
        return gif

    mp4 = out.with_suffix(".mp4")
    try:
        writer = imageio.get_writer(
            mp4, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        try:
            for f in frames:
                writer.append_data(f)
        finally:
            writer.close()
        return mp4
    except Exception as exc:   # pragma: no cover - fall back when ffmpeg missing
        print(f"  [warn] MP4 encoding failed ({exc}); writing GIF instead.")
        gif = out.with_suffix(".gif")
        imageio.mimsave(gif, frames, fps=fps, loop=0)
        return gif
