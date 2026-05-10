import logging
import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QWidget

from gridplayer.pva_player.pva_subscriber import PVASubscriber
from gridplayer.vlc_player.static import (
    DISABLED_TRACK,
    AudioTrack,
    Media,
    MediaInput,
    VideoTrack,
)


def _numpy_to_qimage(arr: np.ndarray) -> QImage | None:
    if arr is None or arr.size == 0:
        return None

    # Normalise non-uint8 to uint8 so QImage can render it
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        # Generic fallback: rescale to 0-255
        amin, amax = float(arr.min()), float(arr.max())
        if amax > amin:
            arr = ((arr - amin) * (255.0 / (amax - amin))).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    arr = np.ascontiguousarray(arr)

    if arr.ndim == 2:
        h, w = arr.shape
        return QImage(
            bytes(arr.tobytes()), w, h, w, QImage.Format_Grayscale8
        ).copy()

    if arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        return QImage(
            bytes(arr.tobytes()), w, h, w * 3, QImage.Format_RGB888
        ).copy()

    if arr.ndim == 3 and arr.shape[2] == 4:
        h, w, _ = arr.shape
        return QImage(
            bytes(arr.tobytes()), w, h, w * 4, QImage.Format_RGBA8888
        ).copy()

    return None


class VideoFramePVA(QWidget):
    """Native EPICS PV Access video driver-widget.

    Connects to a PVA NTNDArray channel via pvapy, decodes frames to QImage,
    and renders them via paintEvent. Bypasses VLC entirely.
    """

    # VideoBlock-facing signals (mirror VLCVideoDriver)
    time_changed = pyqtSignal(int)
    playback_status_changed = pyqtSignal(bool)
    video_ready = pyqtSignal()
    error = pyqtSignal(str)
    crash = pyqtSignal(str)
    update_status = pyqtSignal(str, int)

    is_opengl = False

    def __init__(self, vlc_options=None, parent=None):
        super().__init__(parent)
        self._log = logging.getLogger(self.__class__.__name__)

        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(pal)

        self._frame: QImage | None = None
        self._is_paused = True
        self._is_initialized = False
        self._media: Media | None = None
        self._media_input: MediaInput | None = None
        self._subscriber: PVASubscriber | None = None
        self._channel_name: str | None = None
        self._start_wallclock_ms: int | None = None

        # The PVA monitor delivers frames asynchronously; we throttle the
        # time_changed signal to ~10 Hz so we don't flood the overlay.
        self._time_timer = QTimer(self)
        self._time_timer.setInterval(100)
        self._time_timer.timeout.connect(self._tick_time)

    # --- VideoBlock contract: load slot ----------------------------------

    def load_video(self, media_input: MediaInput):
        self._media_input = media_input
        uri = media_input.uri
        if not uri.startswith("pva://"):
            self.error.emit(f"Not a PVA URI: {uri}")
            return

        self._channel_name = uri[len("pva://") :]
        self._subscriber = PVASubscriber(self._channel_name, parent=self)
        self._subscriber.frame_received.connect(self._on_frame)
        self._subscriber.error.connect(self._on_error)
        self._subscriber.connected.connect(self._on_connected)

        self.update_status.emit(f"Subscribing to {self._channel_name}", 0)
        self._subscriber.start()

    # --- subscriber handlers ---------------------------------------------

    def _on_connected(self):
        # Synthesize a Media object so VideoBlock's load_video_finish has the
        # shape it expects.
        video_track = VideoTrack(
            codec="EPICS NTNDArray",
            bitrate=0,
            language=None,
            description=self._channel_name,
            video_dimensions=(0, 0),
            fps=None,
        )
        self._media = Media(
            length=-1,  # live stream
            video_tracks={0: video_track},
            audio_tracks={},
            cur_video_track_id=0,
            cur_audio_track_id=DISABLED_TRACK,
        )
        self._is_initialized = True
        self._is_paused = False
        self._start_wallclock_ms = int(time.monotonic() * 1000)
        self._time_timer.start()
        self.video_ready.emit()
        self.playback_status_changed.emit(False)

    def _on_error(self, msg: str):
        self.error.emit(msg)

    def _on_frame(self, arr):
        if self._is_paused:
            return
        img = _numpy_to_qimage(arr)
        if img is None:
            return
        self._frame = img
        # Update the synthesized track's dimensions on the first frame so the
        # info overlay shows something useful.
        if self._media and self._media.video_tracks:
            track = self._media.video_tracks[0]
            if track.video_dimensions == (0, 0):
                track.video_dimensions = (img.width(), img.height())
        self.update()

    def _tick_time(self):
        if self._is_paused or self._start_wallclock_ms is None:
            return
        elapsed = int(time.monotonic() * 1000) - self._start_wallclock_ms
        self.time_changed.emit(elapsed)

    # --- paint -----------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._frame is None:
            return
        # Letterbox to fit while preserving aspect
        scaled = self._frame.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawImage(x, y, scaled)

    # --- VideoBlock-facing properties ------------------------------------

    @property
    def is_video_initialized(self) -> bool:
        return self._is_initialized

    @property
    def is_live(self) -> bool:
        return True

    @property
    def length(self) -> int:
        return -1

    @property
    def video_tracks(self):
        return self._media.video_tracks if self._media else {}

    @property
    def audio_tracks(self):
        return {}

    @property
    def cur_video_track_id(self):
        return 0 if self._is_initialized else None

    @property
    def cur_audio_track_id(self):
        return DISABLED_TRACK

    @property
    def media(self):
        return self._media

    # --- VideoBlock-facing methods (mostly no-ops for live PVA) ----------

    def cleanup(self):
        self._time_timer.stop()
        if self._subscriber is not None:
            try:
                self._subscriber.frame_received.disconnect()
                self._subscriber.error.disconnect()
                self._subscriber.connected.disconnect()
            except Exception:
                pass
            self._subscriber.stop()
            self._subscriber.deleteLater()
            self._subscriber = None

    def play(self):
        if not self._is_paused:
            return
        self._is_paused = False
        self.playback_status_changed.emit(False)

    def set_pause(self, is_paused: bool):
        if self._is_paused == is_paused:
            return
        self._is_paused = is_paused
        self.playback_status_changed.emit(is_paused)

    def set_time(self, seek_ms: int):
        # Live stream: no seek
        pass

    def set_playback_rate(self, rate: float):
        # Live stream: no rate control
        pass

    def audio_set_mute(self, is_muted: bool):
        pass

    def audio_set_volume(self, volume: float):
        pass

    def set_video_track(self, track_id: int):
        pass

    def set_audio_track(self, track_id: int):
        pass

    def set_audio_channel_mode(self, mode):
        pass

    def set_aspect_ratio(self, aspect):
        # Aspect handled in paintEvent via KeepAspectRatio; trigger repaint.
        self.update()

    def set_scale(self, scale, *args, **kwargs):
        self.update()

    def set_crop(self, crop, *args, **kwargs):
        self.update()

    def adjust_view(self, *args, **kwargs):
        self.update()

    def get_ms_per_frame(self) -> int:
        # Unknown for an EPICS stream; default to ~30 FPS
        return 33

    def snapshot(self):
        # Not implemented for PVA streams; emit no-op signal if listened.
        pass
