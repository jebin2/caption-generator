from jebin_lib import load_env, utils
load_env()

import os
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from custom_logger import logger_config
import threading
import time
from browser_manager.browser_config import BrowserConfig

_log_lock = Lock()

def _log(level, msg):
    with _log_lock:
        getattr(logger_config, level)(msg)


class HandlerSkippedException(Exception):
    pass


class MultiTypeCaptionGenerator:
    def __init__(self, cache_path, sources, FYI="", skip_duration_seconds=100):
        self.cache_path = cache_path
        self.sources = sources
        self.num_types = len(self.sources) + 1
        self.lock = Lock()
        self.handler_lock = Lock()
        self.FYI = FYI
        self.skip_duration = skip_duration_seconds
        self._thread_local = threading.local()
        self.handler_statuses = {
            i: {"is_skipped": False, "skip_until": 0, "failure_count": 0}
            for i in range(len(self.sources))
        }

    def _load_temp(self, temp_path):
        if utils.path_exists(temp_path):
            with open(temp_path, "r") as f:
                try:
                    return json.load(f)
                except Exception:
                    pass
        return [{"in_progress": False, "processed": False, "scene_caption": None, "scene_dialogue": None}
                for _ in range(self.num_frames)]

    def _save_temp(self, temp_path, data):
        with self.lock:
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

    def _get_next_index(self, temp_path):
        with self.lock:
            temp_data = self._load_temp(temp_path)
            timeout = 60 * 10

            released = False
            for i, entry in enumerate(temp_data):
                if entry["in_progress"]:
                    if entry.get("progress_start_time") and (time.time() - entry["progress_start_time"]) > timeout:
                        entry["in_progress"] = False
                        entry["progress_start_time"] = None
                        released = True
                        _log('warning', f"[Index {i}] Releasing stale in-progress frame")

            if released:
                with open(temp_path, "w") as f:
                    json.dump(temp_data, f, indent=4, ensure_ascii=False)

            for i, entry in enumerate(temp_data):
                if not entry["in_progress"] and not entry["processed"]:
                    temp_data[i]["in_progress"] = True
                    temp_data[i]["progress_start_time"] = time.time()
                    with open(temp_path, "w") as f:
                        json.dump(temp_data, f, indent=4, ensure_ascii=False)
                    return i, temp_data
            return None, temp_data

    def _worker(self, prompt, extract_scenes_json, temp_path, type_id):
        worker_processed_count = 0
        thread_id = threading.current_thread().ident
        handler_key = (type_id - 1) % len(self.sources)
        _log('info', f"[W{type_id}|H{handler_key}] Started on thread {thread_id}")

        while True:
            skip_time = None
            with self.handler_lock:
                status = self.handler_statuses[handler_key]
                if status["is_skipped"]:
                    remaining = status["skip_until"] - time.time()
                    if remaining > 0:
                        skip_time = remaining
                    else:
                        _log('info', f"[H{handler_key}] Reactivating after skip period")
                        status["is_skipped"] = False
                        status["failure_count"] = 0

            if skip_time:
                _log('warning', f"[W{type_id}|H{handler_key}] Paused — handler skipped for {skip_time:.0f}s more")
                time.sleep(skip_time + 1)
                continue

            result_tuple = self._get_next_index(temp_path)
            if result_tuple[0] is None:
                with self.lock:
                    temp_data = self._load_temp(temp_path)
                if all(entry["processed"] for entry in temp_data):
                    _log('success', f"[W{type_id}] All frames processed. Exiting.")
                    break
                else:
                    time.sleep(5)
                    continue

            idx, temp_data = result_tuple
            scene = extract_scenes_json[idx]
            frame_path = scene["frame_path"][0]
            dialogue = scene.get("scene_dialogue", None)
            _log('info', f"[W{type_id}] Processing frame {idx+1}/{len(extract_scenes_json)}")

            try:
                new_prompt = (
                    f"{prompt} Also identify all the characters name in this frame. "
                    f"Keep your description to exactly 100 words or fewer.\n{self.FYI}"
                )
                result = self.search_in_ui_type(type_id, new_prompt, frame_path, thread_id)

                with self.lock:
                    temp_data = self._load_temp(temp_path)
                    if result:
                        temp_data[idx]["scene_caption"] = result.lower()
                        temp_data[idx]["processed"] = True
                        temp_data[idx]["scene_dialogue"] = dialogue
                        temp_data[idx]["frame_path"] = frame_path
                        worker_processed_count += 1
                        _log('success', f"[W{type_id}] Frame {idx+1} done")
                    else:
                        temp_data[idx]["processed"] = False
                        _log('error', f"[W{type_id}] Frame {idx+1} returned no result")
                    temp_data[idx]["in_progress"] = False
                    with open(temp_path, "w") as f:
                        json.dump(temp_data, f, indent=4, ensure_ascii=False)

            except HandlerSkippedException:
                _log('warning', f"[W{type_id}] Handler skipped — releasing frame {idx}")
                with self.lock:
                    temp_data = self._load_temp(temp_path)
                    temp_data[idx]["in_progress"] = False
                    with open(temp_path, "w") as f:
                        json.dump(temp_data, f, indent=4, ensure_ascii=False)
                time.sleep(5)

            except Exception as e:
                _log('error', f"[W{type_id}] Error on frame {idx}: {e}")
                with self.lock:
                    temp_data = self._load_temp(temp_path)
                    temp_data[idx]["in_progress"] = False
                    temp_data[idx]["processed"] = False
                    with open(temp_path, "w") as f:
                        json.dump(temp_data, f, indent=4, ensure_ascii=False)

        if hasattr(self._thread_local, 'handler') and self._thread_local.handler is not None:
            try:
                self._thread_local.handler.cleanup()
            except Exception:
                pass
            self._thread_local.handler = None

        _log('info', f"[W{type_id}] Finished — processed {worker_processed_count} frames")

    def caption_generation(self, extract_scenes_json):
        self.num_frames = len(extract_scenes_json)
        partial_dir = os.path.join(self.cache_path, "partial_captions")
        utils.create_directory(partial_dir)
        temp_path = os.path.join(partial_dir, "temp_progress.json")

        if len([s for s in extract_scenes_json if not s.get("scene_caption", "")]) == 0:
            return extract_scenes_json

        _log('info', f"Starting caption generation for {len(extract_scenes_json)} frames")

        if not utils.path_exists(temp_path):
            self._save_temp(temp_path, [
                {"in_progress": False, "processed": False, "scene_caption": None,
                 "scene_dialogue": extract_scenes_json[i].get("scene_dialogue"), "frame_path": extract_scenes_json[i]["frame_path"][0], "progress_start_time": None}
                for i in range(self.num_frames)
            ])
        else:
            temp_data = self._load_temp(temp_path)
            if len(temp_data) != self.num_frames:
                _log('warning', f"Temp file length mismatch — reinitializing")
                self._save_temp(temp_path, [
                    {"in_progress": False, "processed": False, "scene_caption": None,
                     "scene_dialogue": extract_scenes_json[i].get("scene_dialogue"), "frame_path": extract_scenes_json[i]["frame_path"][0]}
                    for i in range(self.num_frames)
                ])
            else:
                completed_count = reset_count = 0
                for i, data in enumerate(temp_data):
                    if data["in_progress"]:
                        data["in_progress"] = False
                        data["processed"] = False
                        data["scene_caption"] = None
                        reset_count += 1
                    if data["processed"] and data["scene_caption"]:
                        completed_count += 1
                    if i < len(extract_scenes_json):
                        data["scene_dialogue"] = extract_scenes_json[i].get("scene_dialogue")
                        data["frame_path"] = extract_scenes_json[i]["frame_path"][0]
                if reset_count > 0:
                    _log('info', f"Reset {reset_count} stale in-progress frames")
                self._save_temp(temp_path, temp_data)
                _log('info', f"Resuming — {completed_count}/{self.num_frames} frames already done")

        prompt = ("Describe what is happening in this video frame as if you're telling a story. "
                  "Focus on the main subjects, their actions, the setting, and any important details.")

        _log('info', f"Launching {self.num_types} worker(s)")
        with ThreadPoolExecutor(max_workers=self.num_types) as executor:
            futures = [
                executor.submit(self._worker, prompt, extract_scenes_json, temp_path, type_id)
                for type_id in range(self.num_types) if type_id != 0
            ]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    _log('error', f"Worker failed: {e}")

        temp_data = self._load_temp(temp_path)
        if any(not cap["processed"] for cap in temp_data):
            raise ValueError("Exited without completing all frames.")

        for entry in temp_data:
            for scene in extract_scenes_json:
                if scene["frame_path"][0] == entry["frame_path"]:
                    scene["scene_caption"] = entry["scene_caption"]
                    break

        _log('success', "All captions generated.")
        return extract_scenes_json

    def search_in_ui_type(self, type_id, prompt, file_path, thread_id):
        import asyncio as _asyncio
        import subprocess as _sp

        handler_key = (type_id - 1) % len(self.sources)

        with self.handler_lock:
            status = self.handler_statuses[handler_key]
            if status["is_skipped"] and time.time() < status["skip_until"]:
                raise HandlerSkippedException(f"Handler {handler_key} is skipped.")
            elif status["is_skipped"]:
                _log('info', f"[H{handler_key}] Reactivating after skip period")
                status["is_skipped"] = False
                status["failure_count"] = 0

        source = self.sources[handler_key]

        try:
            if not hasattr(self._thread_local, 'handler'):
                self._thread_local.handler = None

            if self._thread_local.handler is None or not isinstance(self._thread_local.handler, source):
                if self._thread_local.handler:
                    try:
                        self._thread_local.handler.cleanup()
                    except Exception:
                        pass
                try:
                    _asyncio.events._set_running_loop(None)
                except Exception:
                    pass

                docker_name = f"thread_id_{thread_id}"
                _sp.run(["docker", "rm", "-f", docker_name], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)

                config = BrowserConfig()
                config.docker_name = docker_name
                neko_file_path = file_path

                if source.__name__ in ("AIStudioUIChat", "QwenUIChat"):
                    neko_base_path = os.path.abspath(os.path.dirname(file_path))
                    neko_file_path = os.path.basename(file_path)
                    config.additionl_docker_flag = ' '.join(utils.get_docker_volume_mounts(config, neko_base_path))

                self._thread_local.handler = source(config=config)

            neko_file_path = file_path
            if source.__name__ in ("AIStudioUIChat", "QwenUIChat"):
                neko_file_path = os.path.basename(file_path)

            result = self._thread_local.handler.chat_fresh(user_prompt=prompt, file_path=neko_file_path)

            if not result or len(result.split(" ")) <= 40:
                raise ValueError(f"Handler {handler_key} returned invalid result.")
            if "AI responses may include mistakes" in result:
                result = result[:result.index("AI responses may include mistakes")]
            if "Sources\nhelp" in result:
                result = result[:result.index("Sources\nhelp\n")]

            with self.handler_lock:
                self.handler_statuses[handler_key]["failure_count"] = 0
            return result

        except Exception as e:
            _log('error', f"[W{type_id}|H{handler_key}] Failed: {e}")
            if hasattr(self._thread_local, 'handler') and self._thread_local.handler:
                try:
                    self._thread_local.handler.cleanup()
                except Exception:
                    pass
                self._thread_local.handler = None
            with self.handler_lock:
                status = self.handler_statuses[handler_key]
                status["failure_count"] += 1
                if status["failure_count"] >= 3:
                    status["is_skipped"] = True
                    status["skip_until"] = time.time() + self.skip_duration
                    _log('warning', f"[H{handler_key}] Failed {status['failure_count']} times — skipping for {self.skip_duration}s")
            raise
