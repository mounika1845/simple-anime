import os
import random
import cv2
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE = os.path.join(SCRIPT_DIR, "krishna.png")

CANVAS_W, CANVAS_H = 1920, 1080
FPS = 60

WINDOW_NAME = "Cute Krishna Reveal"
FULLSCREEN = True

DOT_REVEAL_SECONDS = 3.5
REFINE_SECONDS = 3.0
HOLD_SECONDS = 3.0
CROSSFADE_SECONDS = 1.5
FINAL_HOLD_SECONDS = 3.0

MOSAIC_BLOCK_SIZE = 10
DARK_PIXEL_SKIP = 18
RANDOM_SEED = 42

EDGE_LOW = 60
EDGE_HIGH = 150
LINE_THICKNESS = 2
GLOW_STRENGTH = 1.6

REFLECTION_HEIGHT_RATIO = 0.28

DISPLAY_BRIGHTNESS = 30
DISPLAY_CONTRAST = 1.0
AUTO_CONTRAST = True


# ============================================================
# LOAD AND FIT IMAGE
# ============================================================

def load_and_fit(path, w, h):

    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(
            f"\nCould not read image:\n{path}\n"
            f"\nMake sure krishna.png is in the same folder as this Python file."
        )

    ih, iw = img.shape[:2]

    scale = min(w / iw, h / ih)

    new_w = int(iw * scale)
    new_h = int(ih * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    x_off = (w - new_w) // 2
    y_off = (h - new_h) // 2

    canvas[
        y_off:y_off + new_h,
        x_off:x_off + new_w
    ] = resized

    return canvas


# ============================================================
# CREATE NEON EDGE IMAGE
# ============================================================

def make_neon_edge_layer(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.bilateralFilter(
        gray,
        7,
        50,
        50
    )

    edges = cv2.Canny(
        gray,
        EDGE_LOW,
        EDGE_HIGH
    )

    if LINE_THICKNESS > 0:

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        edges = cv2.dilate(
            edges,
            kernel,
            iterations=LINE_THICKNESS
        )

    mask = edges > 0

    # --------------------------------------------------------
    # Boost image colors
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv[..., 1] = np.clip(
        hsv[..., 1] * 1.7,
        0,
        255
    )

    hsv[..., 2] = np.clip(
        hsv[..., 2] * 1.5 + 60,
        0,
        255
    )

    boosted = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    color_layer = np.zeros_like(img)

    color_layer[mask] = boosted[mask]

    # --------------------------------------------------------
    # Glow
    # --------------------------------------------------------

    inner_glow = cv2.GaussianBlur(
        color_layer,
        (0, 0),
        sigmaX=4
    )

    outer_glow = cv2.GaussianBlur(
        color_layer,
        (0, 0),
        sigmaX=14
    )

    combined = (
        color_layer.astype(np.float32) * 1.4
        + inner_glow.astype(np.float32) * 1.1
        + outer_glow.astype(np.float32) * 0.7
    )

    combined *= GLOW_STRENGTH

    combined = np.clip(
        combined,
        0,
        255
    )

    # --------------------------------------------------------
    # Auto contrast
    # --------------------------------------------------------

    if AUTO_CONTRAST:

        lit_pixels = combined[combined > 5]

        if lit_pixels.size:

            ref = np.percentile(
                lit_pixels,
                99
            )

            if ref > 1:

                combined *= (
                    255.0 / ref
                )

                combined = np.clip(
                    combined,
                    0,
                    255
                )

    return combined.astype(np.uint8)


# ============================================================
# FAST MOSAIC
# ============================================================

def block_mosaic(img, block_size):

    if block_size <= 1:
        return img.copy()

    h, w = img.shape[:2]

    pad_h = (
        block_size - h % block_size
    ) % block_size

    pad_w = (
        block_size - w % block_size
    ) % block_size

    padded = np.pad(
        img,
        (
            (0, pad_h),
            (0, pad_w),
            (0, 0)
        ),
        mode="constant"
    )

    H, W = padded.shape[:2]

    gh = H // block_size
    gw = W // block_size

    reshaped = padded.reshape(
        gh,
        block_size,
        gw,
        block_size,
        3
    )

    reshaped = reshaped.transpose(
        0,
        2,
        1,
        3,
        4
    )

    reshaped = reshaped.reshape(
        gh,
        gw,
        block_size * block_size,
        3
    )

    # Find brightest pixel in each block
    intensity = reshaped.sum(axis=2)

    idx = np.argmax(
        intensity,
        axis=2
    )

    rows = np.arange(gh)[:, None]
    cols = np.arange(gw)[None, :]

    block_colors = reshaped[
        rows,
        cols,
        idx
    ]

    result = np.repeat(
        np.repeat(
            block_colors,
            block_size,
            axis=0
        ),
        block_size,
        axis=1
    )

    return result[:h, :w].astype(np.uint8)


# ============================================================
# CREATE BLOCK LIST
# ============================================================

def get_block_grid(mosaic_img, block_size):

    h, w = mosaic_img.shape[:2]

    blocks = []

    for by in range(0, h, block_size):

        for bx in range(0, w, block_size):

            color = mosaic_img[by, bx]

            if int(color.max()) <= DARK_PIXEL_SKIP:
                continue

            bw = min(
                block_size,
                w - bx
            )

            bh = min(
                block_size,
                h - by
            )

            blocks.append(
                (
                    bx,
                    by,
                    bw,
                    bh,
                    tuple(int(c) for c in color)
                )
            )

    return blocks


# ============================================================
# DRAW DOT
# ============================================================

def draw_block(canvas, block):

    bx, by, bw, bh, color = block

    center = (
        bx + bw // 2,
        by + bh // 2
    )

    radius = max(
        1,
        min(bw, bh) // 2
    )

    cv2.circle(
        canvas,
        center,
        radius,
        color,
        -1,
        lineType=cv2.LINE_AA
    )


# ============================================================
# REFLECTION
# ============================================================

def add_reflection(frame, ratio):

    h, w = frame.shape[:2]

    refl_h = int(
        h * ratio
    )

    if refl_h <= 0:
        return frame.copy()

    source = frame[
        h - refl_h:h
    ]

    reflection = cv2.flip(
        source,
        0
    )

    fade = np.linspace(
        0.35,
        0.0,
        refl_h,
        dtype=np.float32
    ).reshape(
        refl_h,
        1,
        1
    )

    reflection = (
        reflection.astype(np.float32)
        * fade
    ).astype(np.uint8)

    result = frame.copy()

    start_y = h - refl_h

    area = result[
        start_y:h
    ]

    blended = cv2.addWeighted(
        area,
        0.4,
        reflection,
        0.9,
        0
    )

    result[
        start_y:h
    ] = np.maximum(
        area,
        blended
    )

    return result


# ============================================================
# SCREEN PREPARATION
# ============================================================

def prepare_for_screen(frame):

    return cv2.convertScaleAbs(
        frame,
        alpha=DISPLAY_CONTRAST,
        beta=DISPLAY_BRIGHTNESS
    )


# ============================================================
# SHOW FRAME
# ============================================================

def show_frame(frame, delay_ms):

    display = prepare_for_screen(frame)

    cv2.imshow(
        WINDOW_NAME,
        display
    )

    key = cv2.waitKey(delay_ms) & 0xFF

    return (
        key != ord("q")
        and key != 27
    )


# ============================================================
# CROSSFADE
# ============================================================

def crossfade(frame_a, frame_b, t):

    return cv2.addWeighted(
        frame_a,
        1.0 - t,
        frame_b,
        t,
        0
    )


# ============================================================
# PRE-CREATE REFLECTION
# ============================================================

def prepare_reflection(frame):

    return add_reflection(
        frame,
        REFLECTION_HEIGHT_RATIO
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading Krishna image...")

    try:

        base = load_and_fit(
            INPUT_IMAGE,
            CANVAS_W,
            CANVAS_H
        )

    except FileNotFoundError as e:

        print(e)
        return

    print("Creating neon effect...")

    neon = make_neon_edge_layer(
        base
    )

    delay_ms = max(
        1,
        int(1000 / FPS)
    )

    # --------------------------------------------------------
    # Prepare sharp neon frame ONCE
    # --------------------------------------------------------

    sharp_frame = prepare_reflection(
        neon
    )

    # --------------------------------------------------------
    # Prepare mosaic ONCE
    # --------------------------------------------------------

    print("Creating mosaic...")

    full_mosaic = block_mosaic(
        neon,
        MOSAIC_BLOCK_SIZE
    )

    blocks = get_block_grid(
        full_mosaic,
        MOSAIC_BLOCK_SIZE
    )

    print(
        f"Reveal blocks: {len(blocks)}"
    )

    # --------------------------------------------------------
    # Random reveal order
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    order = blocks[:]

    rng.shuffle(order)

    # --------------------------------------------------------
    # WINDOW
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    if FULLSCREEN:

        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    # ========================================================
    # 1. DOT REVEAL
    # ========================================================

    print("1/5  Dot reveal...")

    dot_frames = max(
        1,
        int(
            DOT_REVEAL_SECONDS * FPS
        )
    )

    canvas = np.zeros_like(
        neon
    )

    revealed = 0

    for i in range(dot_frames):

        target = int(
            len(order)
            * (i + 1)
            / dot_frames
        )

        for block in order[
            revealed:target
        ]:

            draw_block(
                canvas,
                block
            )

        revealed = target

        frame = prepare_reflection(
            canvas
        )

        if not show_frame(
            frame,
            delay_ms
        ):
            return

    # ========================================================
    # 2. REFINE
    # ========================================================

    print("2/5  Refining...")

    refine_frames = max(
        1,
        int(
            REFINE_SECONDS * FPS
        )
    )

    # Pre-create refinement frames.
    # This prevents expensive mosaic calculations
    # while the animation is playing.

    refine_frames_cache = []

    for i in range(refine_frames):

        t = (
            i + 1
        ) / refine_frames

        block_size = max(
            1,
            int(
                MOSAIC_BLOCK_SIZE
                * (1 - t) ** 2
            )
        )

        refined = block_mosaic(
            neon,
            block_size
        )

        refined = prepare_reflection(
            refined
        )

        refine_frames_cache.append(
            refined
        )

    for frame in refine_frames_cache:

        if not show_frame(
            frame,
            delay_ms
        ):
            return

    # ========================================================
    # 3. HOLD NEON
    # ========================================================

    print("3/5  Holding neon...")

    hold_frames = max(
        1,
        int(
            HOLD_SECONDS * FPS
        )
    )

    for _ in range(
        hold_frames
    ):

        if not show_frame(
            sharp_frame,
            delay_ms
        ):
            return

    # ========================================================
    # 4. CROSSFADE
    # ========================================================

    print("4/5  Crossfading...")

    crossfade_frames = max(
        1,
        int(
            CROSSFADE_SECONDS * FPS
        )
    )

    for i in range(
        crossfade_frames
    ):

        t = (
            i + 1
        ) / crossfade_frames

        frame = crossfade(
            sharp_frame,
            base,
            t
        )

        if not show_frame(
            frame,
            delay_ms
        ):
            return

    # ========================================================
    # 5. FINAL IMAGE
    # ========================================================

    print("5/5  Final Krishna...")

    final_hold_frames = max(
        1,
        int(
            FINAL_HOLD_SECONDS * FPS
        )
    )

    for _ in range(
        final_hold_frames
    ):

        if not show_frame(
            base,
            delay_ms
        ):
            return

    # ========================================================
    # CLEANUP
    # ========================================================

    cv2.destroyAllWindows()

    print("Animation completed!")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()