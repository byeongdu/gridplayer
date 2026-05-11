import logging
import threading

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


PVA_VALUE_FIELDS = (
    ("booleanValue", np.bool_),
    ("byteValue", np.int8),
    ("ubyteValue", np.uint8),
    ("shortValue", np.int16),
    ("ushortValue", np.uint16),
    ("intValue", np.int32),
    ("uintValue", np.uint32),
    ("longValue", np.int64),
    ("ulongValue", np.uint64),
    ("floatValue", np.float32),
    ("doubleValue", np.float64),
)


def _extract_ntndarray(pv) -> np.ndarray | None:
    """Convert an NTNDArray PvObject to a NumPy array shaped (h, w) or (h, w, c)."""
    try:
        value_union = pv["value"][0]
    except Exception:
        return None

    flat = None
    dtype = None
    for field_name, np_dtype in PVA_VALUE_FIELDS:
        if field_name in value_union:
            raw = value_union[field_name]
            if raw is None or len(raw) == 0:
                continue
            flat = np.frombuffer(bytes(raw), dtype=np_dtype) if isinstance(
                raw, (bytes, bytearray, memoryview)
            ) else np.asarray(raw, dtype=np_dtype)
            dtype = np_dtype
            break

    if flat is None:
        return None

    try:
        dims = [d["size"] for d in pv["dimension"]]
    except Exception:
        dims = []

    if not dims:
        return None

    # PVA dimension order is [width, height] or [width, height, channels].
    # NumPy expects (height, width[, channels]).
    if len(dims) == 1:
        return flat.reshape(dims[0])
    if len(dims) == 2:
        w, h = dims
        return flat.reshape((h, w))
    if len(dims) == 3:
        a, b, c = dims
        # Channels-first: [colour, width, height] (some areaDetector configs)
        if a in (1, 3, 4) and a < b and a < c:
            return flat.reshape((c, b, a))
        # Channels-last: [width, height, colour]
        return flat.reshape((b, a, c))

    return flat.reshape(dims[::-1])


class PVASubscriber(QObject):
    """Subscribes to an EPICS PV Access NTNDArray channel.

    The pvaccess monitor callback fires on a pvapy-internal thread; we keep the
    callback minimal (extract NumPy, emit signal). Qt marshals the signal to the
    receiver's thread via auto-connection, so QImage construction can happen on
    the GUI thread.
    """

    frame_received = pyqtSignal(object)
    error = pyqtSignal(str)
    connected = pyqtSignal()

    def __init__(self, channel_name: str, parent=None):
        super().__init__(parent)
        self._log = logging.getLogger(self.__class__.__name__)
        self._channel_name = channel_name
        self._channel = None
        self._lock = threading.Lock()
        self._is_running = False

    def start(self):
        with self._lock:
            if self._is_running:
                return
            try:
                import pvaccess as pva
            except ImportError as e:
                self.error.emit(
                    "EPICS PVA support requires the 'pvapy' package. "
                    f"Install with: pip install pvapy ({e})"
                )
                return

            try:
                self._channel = pva.Channel(self._channel_name, pva.PVA)
                print(f"DEBUG: Created PVA channel for {self._channel_name}")
            except Exception as e:
                print(f"DEBUG: Failed to create PVA channel: {e}")
                self._log.exception("Failed to create PVA channel")
                self.error.emit(f"Failed to create channel for {self._channel_name}: {e}")
                self._channel = None
                return

            try:
                self._channel.subscribe("gridplayer", self._on_frame)
                print(f"DEBUG: Subscribed to PVA channel {self._channel_name}")
            except Exception as e:
                print(f"DEBUG: Failed to subscribe to PVA channel: {e}")
                self._log.exception("Failed to subscribe to PVA channel")
                self.error.emit(f"Failed to subscribe to {self._channel_name}: {e}")
                self._channel = None
                return

            try:
                self._channel.startMonitor()
                print(f"DEBUG: Started monitor on PVA channel {self._channel_name}")
            except Exception as e:
                print(f"DEBUG: Failed to start monitor on PVA channel: {e}")
                self._log.exception("Failed to start PVA monitor")
                self.error.emit(f"Failed to start monitor on {self._channel_name}: {e}")
                self._channel = None
                return

            self._is_running = True
            print(f"DEBUG: PVA subscriber started for {self._channel_name}")
            self.connected.emit()

    def stop(self):
        with self._lock:
            if not self._is_running:
                return
            try:
                self._channel.stopMonitor()
                self._channel.unsubscribe("gridplayer")
            except Exception:
                self._log.exception("Error stopping PVA monitor")
            finally:
                self._channel = None
                self._is_running = False

    def _on_frame(self, pv):
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        try:
            arr = _extract_ntndarray(pv)
        except Exception:
            self._log.exception("Failed to extract NTNDArray")
            return

        if arr is None:
            return

        # Force a contiguous copy: the PvObject buffer is freed when the
        # callback returns, and NumPy views into it would dangle.
        self.frame_received.emit(np.ascontiguousarray(arr).copy())
