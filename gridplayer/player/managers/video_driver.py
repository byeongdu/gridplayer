from functools import partial

from gridplayer.exceptions import PlayerException
from gridplayer.params import env
from gridplayer.params.static import VideoDriver
from gridplayer.player.managers.base import ManagerBase
from gridplayer.settings import Settings
from gridplayer.widgets.video_frame_dummy import VideoFrameDummy


class VideoDriverManager(ManagerBase):
    # Note: VLC related classes are imported lazily inside `video_driver()`
    _multiprocess_drivers = {
        VideoDriver.VLC_SW,
        VideoDriver.VLC_HW,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._ctx.video_driver = self.video_driver

        self._process_manager = None

    def video_driver(self, uri: str | None = None):
        if uri and uri.startswith("pva://"):
            try:
                from gridplayer.widgets.video_frame_pva import VideoFramePVA
            except ImportError:
                self._log.error(
                    "EPICS PVA support requires the 'pvapy' package."
                    " Install with: pip install pvapy"
                )
                return VideoFrameDummy
            return VideoFramePVA

        video_driver = Settings().get("player/video_driver")

        if video_driver == VideoDriver.VLC_HW and env.IS_MACOS:
            video_driver = VideoDriver.VLC_HW_SP
            Settings().set("player/video_driver", video_driver)

        is_multiprocess = video_driver in self._multiprocess_drivers

        # Dummy driver is always available
        if video_driver == VideoDriver.DUMMY:
            return VideoFrameDummy

        # For VLC drivers, import implementations lazily so the app can run
        # even when VLC libs are missing (e.g., testing with DUMMY driver)
        try:
            if is_multiprocess:
                from gridplayer.vlc_player.instance import ProcessManagerVLC
                from gridplayer.widgets.video_frame_vlc_sw import InstanceProcessVLCSW, VideoFrameVLCSW
                from gridplayer.widgets.video_frame_vlc_hw import InstanceProcessVLCHW, VideoFrameVLCHW

                process_map = {
                    VideoDriver.VLC_SW: InstanceProcessVLCSW,
                    VideoDriver.VLC_HW: InstanceProcessVLCHW,
                }

                driver_map = {
                    VideoDriver.VLC_SW: VideoFrameVLCSW,
                    VideoDriver.VLC_HW: VideoFrameVLCHW,
                }

                if self._process_manager is None:
                    self._process_manager = ProcessManagerVLC(process_map[video_driver])
                    self._process_manager.crash.connect(self.crash)

                return partial(driver_map[video_driver], process_manager=self._process_manager)
            else:
                from gridplayer.widgets.video_frame_vlc_hw_sp import VideoFrameVLCHWSP

                driver_map = {
                    VideoDriver.VLC_HW_SP: VideoFrameVLCHWSP,
                }

                return driver_map[video_driver]
        except Exception:
            # Fall back to dummy if VLC-related imports fail
            return VideoFrameDummy

    def cleanup(self):
        if self._process_manager:
            self._process_manager.cleanup()
            self._process_manager = None

    def set_log_level_vlc(self, log_level):
        if self._process_manager:
            self._process_manager.set_log_level_vlc(log_level)
        elif Settings().get("player/video_driver") == VideoDriver.VLC_HW_SP:
            for vb in self._ctx.video_blocks:
                vb.video_driver.set_log_level_vlc(log_level)

    def set_log_level(self, log_level):
        if self._process_manager:
            self._process_manager.set_log_level(log_level)

    def crash(self, traceback_txt):
        raise PlayerException(traceback_txt)
