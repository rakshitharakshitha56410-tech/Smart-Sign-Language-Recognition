"""
ASL SYSTEM v9.0 — TYPE-TO-SIGN AVATAR
════════════════════════════════════════════════════════════════
Type a word or a single letter, hit ENTER, and the illustrated
avatar (from text_to_sign.py) signs it back — live, in one window,
no camera needed for this direction.

Layout (matches the two-panel mock):

    ┌─────────────────────────────────┬───────────────┐
    │  avatar signing output           │      (o)      │  <- status icon
    │                                   │  ┌─────────┐  │
    │            [ AVATAR ]             │  │ letters │  │  <- text input
    │                                   │  │  or...  │  │
    │                                   │  └─────────┘  │
    └─────────────────────────────────┴───────────────┘

The status icon is a static ring while idle and spins while the
avatar is actively signing, so it reads as a busy/processing
indicator rather than decoration.

Unlike EnglishToSignEngine.play(), typed text is signed literally
(no GlossConverter sentence-reduction) — multi-word input is signed
token by token, word lookup first, fingerspelled fallback per token.

Controls
--------
  a-z, space    type
  ENTER         sign what's typed, then clear the box
  BACKSPACE     delete last character
  TAB           toggle: word lookup <-> force fingerspell
  ESC / Q       quit

Run directly:
    python type_to_sign.py

Or use programmatically:
    from type_to_sign import TypeToSignApp
    TypeToSignApp().run()
"""

import math
import time
import cv2
import numpy as np

from text_to_sign import SignLibrary, AvatarRenderer, SignPlayer

WIN_NAME = "ASL — Type to Sign"

# ── window / layout geometry (proportions match the mock) ─────────────
WIN_W, WIN_H = 1000, 460
DIVIDER_X = 680                       # left avatar panel : right input panel
LEFT_W = DIVIDER_X
RIGHT_W = WIN_W - DIVIDER_X

AVATAR_W, AVATAR_H = 340, 400         # rendered avatar size within left panel
AVATAR_TOP_PAD = 40                   # room for the "avatar signing output" label

ICON_CENTER = (DIVIDER_X + RIGHT_W // 2, 62)
ICON_R = 22

BOX_W, BOX_H = 210, 92
BOX_X1 = DIVIDER_X + (RIGHT_W - BOX_W) // 2
BOX_Y1 = 100
BOX_X2, BOX_Y2 = BOX_X1 + BOX_W, BOX_Y1 + BOX_H

# ── palette (matches the v9.0 app's light UI) ──────────────────────────
BG = (250, 249, 248)
PANEL = (244, 242, 240)
BORDER = (210, 207, 203)
BORDER_MED = (185, 181, 175)
TEXT_PRIMARY = (30, 28, 26)
TEXT_LIGHT = (160, 156, 150)
TEXT_SECONDARY = (100, 96, 90)
BLUE = (35, 120, 205)
GREEN = (80, 160, 70)
RED = (55, 50, 195)
GRAY = (140, 136, 130)


class TypeToSignApp:
    def __init__(self, library: SignLibrary = None):
        self.library = library or SignLibrary()
        self.renderer = AvatarRenderer(size=(AVATAR_W, AVATAR_H))
        self.player = SignPlayer(self.library, renderer=self.renderer)
        self.typed = ""
        self.force_fingerspell = False
        self.status = ""
        self.status_col = TEXT_LIGHT
        self._icon_angle = 0.0
        self._busy = False

    # ---- chrome shared by every frame ------------------------------
    def _base_canvas(self) -> np.ndarray:
        canvas = np.full((WIN_H, WIN_W, 3), BG, np.uint8)
        cv2.rectangle(canvas, (1, 1), (WIN_W - 2, WIN_H - 2), BORDER_MED, 2)
        cv2.line(canvas, (DIVIDER_X, 0), (DIVIDER_X, WIN_H), BORDER_MED, 2)
        cv2.putText(canvas, "avatar signing output", (18, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_SECONDARY, 1, cv2.LINE_AA)
        return canvas

    def _place_avatar(self, canvas: np.ndarray, avatar_img: np.ndarray):
        x = (LEFT_W - AVATAR_W) // 2
        y = AVATAR_TOP_PAD
        canvas[y:y + AVATAR_H, x:x + AVATAR_W] = avatar_img

    def _draw_icon(self, canvas: np.ndarray):
        cx, cy = ICON_CENTER
        col = BLUE if self._busy else BORDER_MED
        if self._busy:
            start = self._icon_angle
            cv2.ellipse(canvas, (cx, cy), (ICON_R, ICON_R), 0,
                        start, start + 300, col, 2, cv2.LINE_AA)
            tip_a = math.radians(start + 300)
            tx = int(cx + ICON_R * math.cos(tip_a))
            ty = int(cy + ICON_R * math.sin(tip_a))
            cv2.circle(canvas, (tx, ty), 3, col, -1, cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (cx, cy), (ICON_R, ICON_R), 0,
                        20, 340, col, 2, cv2.LINE_AA)
            tx = int(cx + ICON_R * math.cos(math.radians(20)))
            ty = int(cy + ICON_R * math.sin(math.radians(20)))
            cv2.circle(canvas, (tx, ty), 3, col, -1, cv2.LINE_AA)

    def _wrap(self, text: str, max_chars: int = 12):
        words = text.split(' ')
        lines, cur = [], ''
        for w in words:
            trial = (cur + ' ' + w).strip()
            if len(trial) > max_chars and cur:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines[:3]

    def _draw_input_box(self, canvas: np.ndarray):
        cv2.rectangle(canvas, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), (255, 255, 255), -1)
        active_col = BLUE if not self.force_fingerspell else GREEN
        cv2.rectangle(canvas, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), active_col, 1)

        cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
        if self.typed:
            lines = self._wrap(self.typed + cursor, max_chars=13)
            col = TEXT_PRIMARY
        else:
            lines = ["letters or", "words input"]
            col = TEXT_LIGHT

        ty = BOX_Y1 + 26
        for line in lines:
            cv2.putText(canvas, line, (BOX_X1 + 12, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
            ty += 22

        mode = "FINGERSPELL" if self.force_fingerspell else "WORD LOOKUP"
        cv2.putText(canvas, f"{mode}  ·  TAB", (BOX_X1, BOX_Y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, GRAY, 1, cv2.LINE_AA)

        if self.status:
            cv2.putText(canvas, self.status[:36], (BOX_X1, BOX_Y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.status_col, 1, cv2.LINE_AA)

    def _compose(self, avatar_img: np.ndarray) -> np.ndarray:
        canvas = self._base_canvas()
        self._place_avatar(canvas, avatar_img)
        self._draw_icon(canvas)
        self._draw_input_box(canvas)
        return canvas

    def _idle_frame(self):
        return self._compose(self.renderer.render(None))

    # ---- signing -----------------------------------------------------
    def _sign_current(self):
        text = self.typed.strip()
        if not text:
            self.status = "Nothing typed yet."
            self.status_col = GRAY
            return

        frames, fingerspelled = self.player.frames_for_text(text, self.force_fingerspell)
        if not frames:
            self.status = f"No sign data for '{text}'."
            self.status_col = RED
            return

        self._busy = True
        delay = max(1, int(1000 / self.player.word_fps))
        for canvas, _ in frames:
            self._icon_angle = (self._icon_angle + 18) % 360
            cv2.imshow(WIN_NAME, self._compose(canvas))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord('q'), 27):
                self._busy = False
                return
        self._busy = False

        self.status = f"Signed: {text.upper()}"
        if fingerspelled:
            self.status = f"Spelled: {', '.join(fingerspelled)}"
        self.status_col = GREEN

    # ---- main loop -----------------------------------------------------
    def run(self):
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
        while True:
            cv2.imshow(WIN_NAME, self._idle_frame())
            key = cv2.waitKey(30) & 0xFF
            if key == 255:  # no key pressed
                continue
            if key in (ord('q'), 27):
                break
            elif key == 13:       # ENTER
                self._sign_current()
                self.typed = ""
            elif key == 8:        # BACKSPACE
                self.typed = self.typed[:-1]
            elif key == 9:        # TAB
                self.force_fingerspell = not self.force_fingerspell
                self.status = ("Fingerspell mode ON" if self.force_fingerspell
                                else "Word lookup mode ON")
                self.status_col = BLUE
            elif key == 32:        # SPACE
                self.typed += " "
            elif 32 < key < 127:   # printable ASCII
                self.typed += chr(key)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    TypeToSignApp().run()