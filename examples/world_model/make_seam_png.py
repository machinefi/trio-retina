"""Render the world-model seam diagram (media/world_model_seam.png).

A tasteful, minimal diagram of the swappable seam:

    any encoder (YOLO · DINOv2 · V-JEPA2)
        -> [ Retina: one standard WorldState ]
            -> any dynamics (imagination rollout)

Pure PIL, brand-clean dark. No model run needed — this is a static design asset.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 360
BG = (14, 16, 22)
INK = (232, 236, 244)
SUBTLE = (138, 146, 162)
MAGENTA = (232, 64, 196)
CARD = (22, 26, 35)
CARD_HI = (30, 35, 47)
STROKE = (52, 58, 72)
MAG_SOFT = (232, 64, 196)


def _font(size: int, bold: bool = False):
    cands = (
        ["/System/Library/Fonts/SFNSDisplay-Bold.otf", "/System/Library/Fonts/HelveticaNeue.ttc"]
        if bold else
        ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/HelveticaNeue.ttc"]
    )
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _round_rect(d, box, r, fill, outline=None, width=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def _ctext(d, cx, y, text, font, fill):
    w = d.textlength(text, font=font)
    d.text((cx - w / 2, y), text, font=font, fill=fill)


def _arrow(d, x1, y, x2, color, width=4):
    d.line([(x1, y), (x2 - 10, y)], fill=color, width=width)
    d.polygon([(x2 - 12, y - 7), (x2, y), (x2 - 12, y + 7)], fill=color)


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f_title = _font(30, bold=True)
    f_card = _font(26, bold=True)
    f_sub = _font(19)
    f_small = _font(17)

    cy = 196
    # --- left card: any encoder ---
    lx0, lx1 = 40, 380
    _round_rect(d, [lx0, cy - 70, lx1, cy + 70], 16, CARD, STROKE, 2)
    _ctext(d, (lx0 + lx1) / 2, cy - 52, "any encoder", f_card, INK)
    _ctext(d, (lx0 + lx1) / 2, cy - 14, "YOLO · DINOv2 · V-JEPA2", f_sub, SUBTLE)
    _ctext(d, (lx0 + lx1) / 2, cy + 18, "(or none — symbolic only)", f_small, SUBTLE)

    # --- center card: Retina WorldState (highlighted) ---
    cx0, cx1 = 470, 810
    _round_rect(d, [cx0, cy - 82, cx1, cy + 82], 18, CARD_HI, MAGENTA, 3)
    _ctext(d, (cx0 + cx1) / 2, cy - 64, "Retina", f_card, MAGENTA)
    _ctext(d, (cx0 + cx1) / 2, cy - 28, "one standard", f_sub, INK)
    _ctext(d, (cx0 + cx1) / 2, cy - 4, "WorldState", _font(28, bold=True), INK)
    _ctext(d, (cx0 + cx1) / 2, cy + 34, "symbolic + latent vec", f_small, SUBTLE)

    # --- right card: any dynamics ---
    rx0, rx1 = 900, 1240
    _round_rect(d, [rx0, cy - 70, rx1, cy + 70], 16, CARD, STROKE, 2)
    _ctext(d, (rx0 + rx1) / 2, cy - 52, "any dynamics", f_card, INK)
    _ctext(d, (rx0 + rx1) / 2, cy - 14, "imagination rollout", f_sub, SUBTLE)
    _ctext(d, (rx0 + rx1) / 2, cy + 18, "(learned world model)", f_small, SUBTLE)

    # --- arrows ---
    _arrow(d, lx1 + 14, cy, cx0 - 14, MAGENTA)
    _arrow(d, cx1 + 14, cy, rx0 - 14, MAGENTA)

    # --- title + caption ---
    d.text((40, 38), "The model-agnostic world-model seam", font=f_title, fill=INK)
    d.text((40, 80),
            "Swap the encoder in front or the dynamics behind — the standardized "
            "state in the middle is the constant.",
            font=f_sub, fill=SUBTLE)

    out = os.path.join(os.path.dirname(__file__), "..", "..", "media", "world_model_seam.png")
    out = os.path.normpath(out)
    img.save(out)
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
