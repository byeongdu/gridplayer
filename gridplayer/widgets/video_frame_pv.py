import logging
import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QWidget

from gridplayer.vlc_player.static import (
    DISABLED_TRACK,
    Media,
    MediaInput,
    VideoTrack,
)

logger = logging.getLogger(__name__)


def _parse_pv_uri(uri: str) -> tuple[str, str, str]:
    """Parse pv://hostname:image_name:field_name into its components."""
    # Strip 'pv://' prefix
    rest = uri[len("pv://") :]
    parts = rest.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"PV URI must be in format pv://hostname:image:field, got: {uri}"
        )
    host, image, field = parts
    return host, image, field


def _build_pv_name(host: str, image: str, field: str) -> str:
    """Build the EPICS PV name: host:image:field."""
    return f"{host}:{image}:{field}"


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


class VideoFramePV(QWidget):
    """EPICS PV video driver-widget using epics.caget().

    Connects to an EPICS PV via pyepics (epics.caget), polls for image data
    on a timer, decodes frames to QImage, and renders via paintEvent.
    Bypasses VLC entirely.

    URI format: pv://hostname:image_name:field_name
    Example:   pv://12idBFS2:image1:ArrayData
    """

    # VideoBlock-facing signals (mirror VLCVideoDriver)
    time_changed = pyqtSignal(int)
    playback_status_changed = pyqtSignal(bool)
    video_ready = pyqtSignal()
    error = pyqtSignal(str)
    crash = pyqtSignal(str)
    update_status = pyqtSignal(str, int)

    is_opengl = False

    # Default poll interval in milliseconds
    DEFAULT_POLL_INTERVAL_MS = 100

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
        self._host: str = ""
        self._image: str = ""
        self._field: str = ""
        self._pv_name: str = ""

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self.DEFAULT_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_frame)

        self._start_wallclock_ms: int | None = None

    # --- VideoBlock contract: load slot ----------------------------------

    def load_video(self, media_input: MediaInput):
        self._media_input = media_input
        uri = media_input.uri
        if not uri.startswith("pv://"):
            self.error.emit(f"Not a PV URI: {uri}")
            return

        try:
            self._host, self._image, self._field = _parse_pv_uri(uri)
        except ValueError as e:
            self.error.emit(str(e))
            return

        self._pv_name = _build_pv_name(self._host, self._image, self._field)

        # Check that epics is available
        try:
            import epics
        except ImportError:
            self.error.emit(
                "EPICS PV support requires the 'pyepics' package. "
                "Install with: pip install pyepics"
            )
            return

        self.update_status.emit(f"Connecting to {self._pv_name}", 0)

        # Try an initial fetch to verify the PV exists and is accessible
        try:
            arr = epics.caget(self._pv_name)
            if arr is None:
                self.error.emit(f"PV {self._pv_name} returned no data")
                return
            self._log.debug(
                f"Initial fetch from {self._pv_name}: "
                f"type={type(arr)}, shape={getattr(arr, 'shape', 'N/A')}"
            )
        except Exception as e:
            self.error.emit(f"Failed to connect to PV {self._pv_name}: {e}")
            return

        # Synthesize a Media object so VideoBlock's load_video_finish has the
        # shape it expects.
        video_track = VideoTrack(
            codec="EPICS PV caget",
            bitrate=0,
            language=None,
            description=self._pv_name,
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
        self._poll_timer.start()
        self.video_ready.emit()
        self.playback_status_changed.emit(False)

    # --- Polling ----------------------------------------------------------

    def _poll_frame(self):
        if self._is_paused:
            return

        try:
            import epics

            arr = epics.caget(self._pv_name)
        except Exception:
            self._log.exception(f"Failed to caget {self._pv_name}")
            return

        if arr is None:
            return

        # epics.caget may return a numpy array or a scalar; ensure ndarray
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        img = _numpy_to_qimage(arr)
        if img is None:
            self._log.warning(
                f"_numpy_to_qimage returned None for array of shape "
                f"{getattr(arr, 'shape', 'unknown')} and dtype "
                f"{getattr(arr, 'dtype', 'unknown')}"
            )
            return

        self._frame = img

        # Update the synthesized track's dimensions on the first frame so the
        # info overlay shows something useful.
        if self._media and self._media.video_tracks:
            track = self._media.video_tracks[0]
            if track.video_dimensions == (0, 0):
                track.video_dimensions = (img.width(), img.height())
        self.update()

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

    # --- VideoBlock-facing methods (mostly no-ops for live PV) ----------

    def cleanup(self):
        self._poll_timer.stop()

    def play(self):
        if not self._is_paused:
            return
        self._is_paused = False
        self._poll_timer.start()
        self.playback_status_changed.emit(False)

    def set_pause(self, is_paused: bool):
        if self._is_paused == is_paused:
            return
        self._is_paused = is_paused
        if is_paused:
            self._poll_timer.stop()
        else:
            self._poll_timer.start()
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

    def set_crosshair(self, enabled: bool):
        pass

    def set_crosshair_position(self, x, y):
        pass

    def set_crosshair_color(self, qcolor):
        pass

    def set_crosshair_thickness(self, thickness):
        pass

    def set_crosshair_full(self, full: bool):
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
        return self.DEFAULT_POLL_INTERVAL_MS

    def snapshot(self):
        # Not implemented for PV streams; emit no-op signal if listened.
        pass