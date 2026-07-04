#!/usr/bin/env python3
"""Maximum-throughput video decode plus YOLO-NAS inference benchmark."""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL = "/app/models/yolo_nas_s.onnx"
DEFAULT_LABELS = "/app/labels/coco-80.txt"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def has_cmd(cmd: str) -> bool:
    return subprocess.run(["sh", "-lc", f"command -v {cmd} >/dev/null 2>&1"]).returncode == 0


def ffmpeg_hwaccels() -> set[str]:
    if not has_cmd("ffmpeg"):
        return set()
    completed = run(["ffmpeg", "-hide_banner", "-hwaccels"], check=False)
    return {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.startswith("Hardware")
    }


def pick_decoder(requested: str) -> tuple[str, list[str]]:
    if requested == "none":
        return ("none", [])
    if requested != "auto":
        if requested == "qsv":
            return ("qsv", ["-hwaccel", "qsv", "-qsv_device", "/dev/dri/renderD128"])
        if requested == "vaapi":
            return ("vaapi", ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"])
        return (requested, ["-hwaccel", requested])

    hwaccels = ffmpeg_hwaccels()
    if has_cmd("nvidia-smi") and "cuda" in hwaccels:
        return ("cuda", ["-hwaccel", "cuda"])
    if Path("/dev/dri").exists() and "vaapi" in hwaccels:
        return ("vaapi", ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"])
    if Path("/dev/dri/renderD128").exists() and "qsv" in hwaccels:
        return ("qsv", ["-hwaccel", "qsv", "-qsv_device", "/dev/dri/renderD128"])
    return ("none", [])


def build_ffmpeg_cmd(
    video: Path,
    loops: int,
    decoder_name: str,
    decoder_args: list[str],
    width: int,
    height: int,
    pipeline: str,
    vf_override: str | None = None,
) -> tuple[list[str], str, int]:
    if vf_override:
        override_decoder_args = decoder_args
        if decoder_name == "cuda":
            override_decoder_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif decoder_name == "vaapi":
            override_decoder_args = [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-hwaccel_device",
                "/dev/dri/renderD128",
            ]
        return (
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-stream_loop",
                str(loops - 1),
                *override_decoder_args,
                "-i",
                str(video),
                "-an",
                "-vf",
                vf_override,
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            "bgr24",
            width * height * 3,
        )

    if pipeline in ("baseline", "letterbox"):
        if pipeline == "letterbox":
            vf = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                "format=bgr24"
            )
        else:
            vf = f"scale={width}:{height},format=bgr24"
        return (
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-stream_loop",
                str(loops - 1),
                *decoder_args,
                "-i",
                str(video),
                "-an",
                "-vf",
                vf,
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            "bgr24",
            width * height * 3,
        )

    if pipeline != "gpu-resize":
        raise SystemExit(f"unsupported pipeline: {pipeline}")
    if decoder_name == "cuda":
        hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf = f"scale_cuda={width}:{height}:interp_algo=bilinear:format=yuv420p,hwdownload,format=yuv420p,format=bgr24"
    elif decoder_name == "vaapi":
        hw_args = [
            "-hwaccel",
            "vaapi",
            "-hwaccel_output_format",
            "vaapi",
            "-hwaccel_device",
            "/dev/dri/renderD128",
        ]
        vf = f"scale_vaapi=w={width}:h={height}:format=yuv420p:mode=hq,hwdownload,format=yuv420p,format=bgr24"
    else:
        raise SystemExit("--pipeline gpu-resize requires --decoder cuda or --decoder vaapi")

    return (
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stream_loop",
            str(loops - 1),
            *hw_args,
            "-i",
            str(video),
            "-an",
            "-vf",
            vf,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        "bgr24",
        width * height * 3,
    )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((pct / 100) * (len(ordered) - 1)))
    return ordered[min(len(ordered) - 1, max(0, index))]


def histogram(values: list[float], buckets: list[float]) -> dict[str, int]:
    out: dict[str, int] = {}
    previous = 0.0
    remaining = list(values)
    for bucket in buckets:
        count = sum(1 for value in remaining if value <= bucket)
        out[f"{previous:.1f}-{bucket:.1f}"] = count
        remaining = [value for value in remaining if value > bucket]
        previous = bucket
    out[f">{buckets[-1]:.1f}"] = len(remaining)
    return out


def ffprobe(video: str) -> dict[str, Any]:
    completed = run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_name,width,height,pix_fmt,avg_frame_rate",
            "-of",
            "json",
            video,
        ],
        check=False,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip()}
    return json.loads(completed.stdout)


def load_labels(path: str) -> dict[int, str]:
    labels = {}
    for index, line in enumerate(Path(path).read_text().splitlines()):
        label = line.strip()
        if label:
            labels[index] = label
    return labels


def read_rapl_energy() -> dict[str, dict[str, Any]]:
    root = Path("/sys/class/powercap")
    out: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return out
    for energy_path in root.glob("intel-rapl*/energy_uj"):
        try:
            out[energy_path.parent.name] = {
                "name": (energy_path.parent / "name").read_text().strip(),
                "energy_uj": int(energy_path.read_text().strip()),
                "max_energy_range_uj": int((energy_path.parent / "max_energy_range_uj").read_text().strip()),
            }
        except Exception:
            continue
    return out


def rapl_delta_w(start: dict[str, dict[str, Any]], end: dict[str, dict[str, Any]], seconds: float) -> dict[str, Any]:
    if seconds <= 0:
        return {}
    out = {}
    for domain, s in start.items():
        e = end.get(domain)
        if not e:
            continue
        delta = int(e["energy_uj"]) - int(s["energy_uj"])
        if delta < 0:
            delta += int(s["max_energy_range_uj"])
        out[domain] = {"name": s["name"], "watts": round((delta / 1_000_000) / seconds, 6)}
    return out


class NvidiaSampler:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.samples: list[dict[str, float | str]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if has_cmd("nvidia-smi"):
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def _run(self) -> None:
        query = (
            "name,power.draw,power.limit,utilization.gpu,utilization.memory,"
            "utilization.decoder,utilization.encoder,memory.used,memory.total,temperature.gpu"
        )
        cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        while not self.stop_event.is_set():
            completed = run(cmd, check=False)
            if completed.returncode == 0:
                for line in completed.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != 10:
                        continue
                    sample: dict[str, float | str] = {"name": parts[0], "ts": time.time()}
                    for key, value in zip(
                        [
                            "power.draw",
                            "power.limit",
                            "utilization.gpu",
                            "utilization.memory",
                            "utilization.decoder",
                            "utilization.encoder",
                            "memory.used",
                            "memory.total",
                            "temperature.gpu",
                        ],
                        parts[1:],
                    ):
                        try:
                            sample[key] = float(value)
                        except ValueError:
                            sample[key] = value
                    self.samples.append(sample)
            self.stop_event.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"available": False, "samples": 0}
        numeric: dict[str, list[float]] = {}
        for sample in self.samples:
            for key, value in sample.items():
                if isinstance(value, float) and key != "ts":
                    numeric.setdefault(key, []).append(value)
        return {
            "available": True,
            "samples": len(self.samples),
            "name": self.samples[-1].get("name"),
            "average": {k: round(statistics.mean(v), 3) for k, v in numeric.items() if v},
            "max": {k: round(max(v), 3) for k, v in numeric.items() if v},
        }


class CpuSampler:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.samples: list[dict[str, float | int]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.child_pid: int | None = None
        self.available = False

    def set_child_pid(self, pid: int) -> None:
        self.child_pid = pid

    def start(self) -> None:
        try:
            import psutil  # noqa: F401
        except Exception:
            return
        self.available = True
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def _run(self) -> None:
        import psutil

        main = psutil.Process()
        tracked: dict[int, Any] = {main.pid: main}
        psutil.cpu_percent(interval=None)
        main.cpu_percent(interval=None)
        while not self.stop_event.is_set():
            if self.child_pid and self.child_pid not in tracked:
                try:
                    child = psutil.Process(self.child_pid)
                    child.cpu_percent(interval=None)
                    tracked[self.child_pid] = child
                except psutil.Error:
                    pass
            sample: dict[str, float | int] = {
                "ts": time.time(),
                "system_percent": psutil.cpu_percent(interval=None),
            }
            total_process = 0.0
            rss = 0
            alive = 0
            for pid, proc in list(tracked.items()):
                try:
                    cpu_percent = proc.cpu_percent(interval=None)
                    total_process += cpu_percent
                    rss += proc.memory_info().rss
                    alive += 1
                    if pid == main.pid:
                        sample["python_percent"] = cpu_percent
                    elif pid == self.child_pid:
                        sample["ffmpeg_percent"] = cpu_percent
                except psutil.Error:
                    tracked.pop(pid, None)
            sample["process_percent"] = total_process
            sample["tracked_processes"] = alive
            sample["rss_mb"] = rss / (1024 * 1024)
            self.samples.append(sample)
            self.stop_event.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "samples": 0}
        if not self.samples:
            return {"available": True, "samples": 0}
        numeric: dict[str, list[float]] = {}
        for sample in self.samples:
            for key, value in sample.items():
                if isinstance(value, (float, int)) and key != "ts":
                    numeric.setdefault(key, []).append(float(value))
        return {
            "available": True,
            "samples": len(self.samples),
            "average": {k: round(statistics.mean(v), 3) for k, v in numeric.items() if v},
            "max": {k: round(max(v), 3) for k, v in numeric.items() if v},
        }


class Backend:
    name: str
    input_name: str
    input_shape: list[int]
    input_dtype: str

    def run(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OnnxBackend(Backend):
    def __init__(self, model: str, providers: list[str]) -> None:
        import onnxruntime as ort

        self.name = "onnx:" + ",".join(providers)
        self.session = ort.InferenceSession(model, providers=providers)
        inp = self.session.get_inputs()[0]
        self.outputs = self.session.get_outputs()
        self.input_name = inp.name
        self.input_shape = [int(v) for v in inp.shape]
        self.input_dtype = inp.type

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]


def parse_ov_property(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_ov_properties(values: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--ov-property must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--ov-property key must not be empty, got: {item}")
        properties[key] = parse_ov_property(value)
    return properties


class OpenVinoBackend(Backend):
    def __init__(self, model: str, device: str, properties: dict[str, Any] | None = None) -> None:
        import openvino as ov

        self.name = f"openvino:{device}"
        self.core = ov.Core()
        self.properties = properties or {}
        self.compiled = self.core.compile_model(model, device, self.properties)
        inp = self.compiled.inputs[0]
        self.input_name = inp.get_any_name()
        self.input_shape = [int(v) for v in inp.get_shape()]
        self.input_dtype = str(inp.get_element_type())
        self.output = self.compiled.outputs[0]

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return self.compiled({self.input_name: tensor})[self.output]

    def create_request(self):
        return self.compiled.create_infer_request()

    def start_async(self, request: Any, tensor: np.ndarray) -> None:
        request.start_async({self.input_name: tensor})

    def wait_async(self, request: Any) -> np.ndarray:
        request.wait()
        tensor = request.get_tensor(self.output)
        return np.array(tensor.data, copy=True)


def pick_backend(requested: str, model: str, ov_properties: dict[str, Any] | None = None) -> Backend:
    if requested == "onnx-cuda":
        return OnnxBackend(model, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    if requested == "onnx-cpu":
        return OnnxBackend(model, ["CPUExecutionProvider"])
    if requested == "openvino-gpu":
        return OpenVinoBackend(model, "GPU", ov_properties)
    if requested == "openvino-npu":
        return OpenVinoBackend(model, "NPU", ov_properties)
    if requested == "openvino-cpu":
        return OpenVinoBackend(model, "CPU", ov_properties)

    if requested != "auto":
        raise SystemExit(f"unsupported backend: {requested}")

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if has_cmd("nvidia-smi") and "CUDAExecutionProvider" in providers:
            return OnnxBackend(model, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception:
        pass

    try:
        import openvino as ov

        devices = ov.Core().available_devices
        if "NPU" in devices:
            return OpenVinoBackend(model, "NPU", ov_properties)
        if "GPU" in devices:
            return OpenVinoBackend(model, "GPU", ov_properties)
    except Exception:
        pass

    try:
        import onnxruntime as ort

        if "CPUExecutionProvider" in ort.get_available_providers():
            return OnnxBackend(model, ["CPUExecutionProvider"])
    except Exception:
        pass

    return OpenVinoBackend(model, "CPU", ov_properties)


def normalize_shape(shape: list[int]) -> tuple[int, int, str]:
    if len(shape) != 4:
        raise SystemExit(f"expected 4D model input, got {shape}")
    if shape[1] == 3:
        return shape[3], shape[2], "nchw"
    if shape[3] == 3:
        return shape[2], shape[1], "nhwc"
    raise SystemExit(f"cannot infer NCHW/NHWC input layout from {shape}")


def dtype_for_input(input_dtype: str) -> np.dtype:
    text = input_dtype.lower()
    if "uint8" in text or "uint8_t" in text:
        return np.dtype(np.uint8)
    if "float16" in text:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


def prepare_tensor(frame_bytes: bytes, width: int, height: int, layout: str, dtype: np.dtype) -> np.ndarray:
    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
    if layout == "nchw":
        frame = np.transpose(frame, (2, 0, 1))[None, ...]
    else:
        frame = frame[None, ...]
    if dtype == np.uint8:
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(frame.astype(dtype) / np.array(255, dtype=dtype))


def parse_yolonas(output: np.ndarray, labels: dict[int, str], threshold: float) -> list[dict[str, Any]]:
    predictions = np.asarray(output)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    detections = []
    for row in predictions:
        if len(row) < 7:
            continue
        _, x_min, y_min, x_max, y_max, confidence, class_id = row[:7]
        class_id_int = int(class_id)
        confidence_float = float(confidence)
        if class_id_int < 0:
            break
        if confidence_float < threshold:
            continue
        detections.append(
            {
                "label": labels.get(class_id_int, str(class_id_int)),
                "class_id": class_id_int,
                "score": confidence_float,
                "box": [float(x_min), float(y_min), float(x_max), float(y_max)],
            }
        )
    return detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "onnx-cuda", "onnx-cpu", "openvino-gpu", "openvino-npu", "openvino-cpu"],
    )
    parser.add_argument("--decoder", default="auto", choices=["auto", "none", "cuda", "qsv", "vaapi"])
    parser.add_argument(
        "--pipeline",
        default="baseline",
        choices=["baseline", "letterbox", "gpu-resize"],
    )
    parser.add_argument("--require-hw-decoder", action="store_true")
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--ffmpeg-vf-override")
    parser.add_argument("--async-requests", type=int, default=1)
    parser.add_argument("--reader-queue", type=int, default=0)
    parser.add_argument(
        "--ov-property",
        action="append",
        default=[],
        help="OpenVINO compile property as KEY=VALUE. Repeat for multiple properties.",
    )
    parser.add_argument("--out")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = Path(args.video)
    model = Path(args.model)
    labels_path = Path(args.labels)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")
    if not labels_path.exists():
        raise SystemExit(f"labels not found: {labels_path}")
    if args.loops < 1:
        raise SystemExit("--loops must be >= 1")
    if args.async_requests < 1:
        raise SystemExit("--async-requests must be >= 1")
    if args.reader_queue < 0:
        raise SystemExit("--reader-queue must be >= 0")

    labels = load_labels(str(labels_path))
    ov_properties = parse_ov_properties(args.ov_property)
    backend = pick_backend(args.backend, str(model), ov_properties)
    width, height, layout = normalize_shape(backend.input_shape)
    input_dtype = dtype_for_input(backend.input_dtype)

    decoder_name, decoder_args = pick_decoder(args.decoder)
    if args.require_hw_decoder and decoder_name == "none":
        raise SystemExit("hardware decoder required, but no supported FFmpeg hwaccel was selected")
    ffmpeg_cmd, output_pixel_format, frame_size = build_ffmpeg_cmd(
        video,
        args.loops,
        decoder_name,
        decoder_args,
        width,
        height,
        args.pipeline,
        args.ffmpeg_vf_override,
    )

    blank = np.zeros(frame_size, dtype=np.uint8).tobytes()
    backend.run(prepare_tensor(blank, width, height, layout, input_dtype))

    nvidia = NvidiaSampler()
    cpu = CpuSampler()
    rapl_start = read_rapl_energy()
    nvidia.start()
    cpu.start()
    start_wall = time.perf_counter()

    process: subprocess.Popen[bytes] | None = None
    ffmpeg_stderr = b""
    return_code = 0

    frames = 0
    latencies: list[float] = []
    step_latencies: dict[str, list[float]] = {
        "read": [],
        "prepare": [],
        "infer": latencies,
        "parse": [],
        "frame_total": [],
    }
    detections_by_label: Counter[str] = Counter()
    frames_by_detection_count: Counter[int] = Counter()
    try:
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cpu.set_child_pid(process.pid)
        assert process.stdout is not None
        frame_queue: queue.Queue[tuple[bytes | None, float]] | None = None
        reader_thread: threading.Thread | None = None

        def read_frame() -> tuple[bytes | None, float]:
            read_start = time.perf_counter()
            raw_frame = process.stdout.read(frame_size)
            return raw_frame, time.perf_counter() - read_start

        if args.reader_queue > 0:
            frame_queue = queue.Queue(maxsize=args.reader_queue)
            reader_pipe_latencies: list[float] = []
            reader_queue_put_latencies: list[float] = []

            def reader() -> None:
                assert frame_queue is not None
                while True:
                    raw_frame, read_seconds = read_frame()
                    reader_pipe_latencies.append(read_seconds)
                    put_start = time.perf_counter()
                    frame_queue.put((raw_frame, read_seconds))
                    reader_queue_put_latencies.append(time.perf_counter() - put_start)
                    if not raw_frame or len(raw_frame) != frame_size:
                        break

            reader_thread = threading.Thread(target=reader, daemon=True)
            reader_thread.start()
            step_latencies["pipe_read"] = reader_pipe_latencies
            step_latencies["queue_put"] = reader_queue_put_latencies
            step_latencies["queue_get"] = []

        def next_frame() -> tuple[bytes | None, float]:
            if frame_queue is None:
                return read_frame()
            get_start = time.perf_counter()
            raw_frame, read_seconds = frame_queue.get()
            step_latencies["queue_get"].append(time.perf_counter() - get_start)
            return raw_frame, read_seconds

        if isinstance(backend, OpenVinoBackend) and args.async_requests > 1:
            requests = [backend.create_request() for _ in range(args.async_requests)]
            free_request_indexes = list(range(args.async_requests))
            slots: list[tuple[int, np.ndarray, float]] = []

            def finish_oldest() -> None:
                nonlocal frames
                request_index, tensor_ref, infer_start = slots.pop(0)
                request = requests[request_index]
                output = backend.wait_async(request)
                free_request_indexes.append(request_index)
                latencies.append(time.perf_counter() - infer_start)
                detections = parse_yolonas(output, labels, args.threshold)
                frames += 1
                frames_by_detection_count[len(detections)] += 1
                for detection in detections:
                    detections_by_label[detection["label"]] += 1

            while True:
                frame_start = time.perf_counter()
                raw, read_seconds = next_frame()
                step_latencies["read"].append(read_seconds)
                if not raw:
                    break
                if len(raw) != frame_size:
                    break
                prepare_start = time.perf_counter()
                if not free_request_indexes:
                    finish_oldest()
                tensor = prepare_tensor(raw, width, height, layout, input_dtype)
                step_latencies["prepare"].append(time.perf_counter() - prepare_start)
                request_index = free_request_indexes.pop(0)
                request = requests[request_index]
                infer_start = time.perf_counter()
                backend.start_async(request, tensor)
                slots.append((request_index, tensor, infer_start))
                if len(slots) >= args.async_requests:
                    finish_oldest()
                step_latencies["frame_total"].append(time.perf_counter() - frame_start)
            while slots:
                finish_oldest()
        else:
            while True:
                frame_start = time.perf_counter()
                raw, read_seconds = next_frame()
                step_latencies["read"].append(read_seconds)
                if not raw:
                    break
                if len(raw) != frame_size:
                    break
                prepare_start = time.perf_counter()
                tensor = prepare_tensor(raw, width, height, layout, input_dtype)
                step_latencies["prepare"].append(time.perf_counter() - prepare_start)
                infer_start = time.perf_counter()
                output = backend.run(tensor)
                latencies.append(time.perf_counter() - infer_start)
                parse_start = time.perf_counter()
                detections = parse_yolonas(output, labels, args.threshold)
                step_latencies["parse"].append(time.perf_counter() - parse_start)
                frames += 1
                frames_by_detection_count[len(detections)] += 1
                for detection in detections:
                    detections_by_label[detection["label"]] += 1
                step_latencies["frame_total"].append(time.perf_counter() - frame_start)
        if reader_thread is not None:
            reader_thread.join(timeout=2)
    finally:
        if process is not None:
            if process.stderr is not None:
                ffmpeg_stderr = process.stderr.read()
            return_code = process.wait(timeout=10)
        elapsed = time.perf_counter() - start_wall
        nvidia.stop()
        cpu.stop()
        rapl_end = read_rapl_energy()

    if return_code != 0 and frames == 0:
        raise SystemExit(ffmpeg_stderr.decode(errors="replace"))

    latency_ms = [v * 1000 for v in latencies]
    step_latency_ms = {name: [v * 1000 for v in values] for name, values in step_latencies.items()}

    def latency_summary(values: list[float]) -> dict[str, Any]:
        return {
            "count": len(values),
            "total_ms": round(sum(values), 3),
            "mean": round(statistics.mean(values), 3) if values else None,
            "min": round(min(values), 3) if values else None,
            "p50": round(percentile(values, 50), 3) if values else None,
            "p95": round(percentile(values, 95), 3) if values else None,
            "p99": round(percentile(values, 99), 3) if values else None,
            "max": round(max(values), 3) if values else None,
        }

    result = {
        "video": str(video),
        "probe": ffprobe(str(video)),
        "loops": args.loops,
        "backend": {
            "requested": args.backend,
            "selected": backend.name,
            "input_name": backend.input_name,
            "input_shape": backend.input_shape,
            "input_dtype": backend.input_dtype,
            "layout": layout,
            "async_requests": args.async_requests,
            "reader_queue": args.reader_queue,
            "ov_properties": ov_properties,
        },
        "decoder": {
            "requested": args.decoder,
            "selected": decoder_name,
            "ffmpeg_args": decoder_args,
            "output_pixel_format": output_pixel_format,
            "pipeline": args.pipeline,
            "command": ffmpeg_cmd,
            "ffmpeg_returncode": return_code,
            "ffmpeg_stderr": ffmpeg_stderr.decode(errors="replace")[-2000:],
        },
        "model": str(model),
        "labels": str(labels_path),
        "threshold": args.threshold,
        "frames": frames,
        "wall_seconds": round(elapsed, 6),
        "end_to_end_fps": round(frames / elapsed, 3) if elapsed else None,
        "inference_fps": round(len(latencies) / sum(latencies), 3) if latencies else None,
        "latency_ms": {
            "mean": round(statistics.mean(latency_ms), 3) if latency_ms else None,
            "min": round(min(latency_ms), 3) if latency_ms else None,
            "p01": round(percentile(latency_ms, 1), 3) if latency_ms else None,
            "p05": round(percentile(latency_ms, 5), 3) if latency_ms else None,
            "p10": round(percentile(latency_ms, 10), 3) if latency_ms else None,
            "p25": round(percentile(latency_ms, 25), 3) if latency_ms else None,
            "p50": round(percentile(latency_ms, 50), 3) if latency_ms else None,
            "p75": round(percentile(latency_ms, 75), 3) if latency_ms else None,
            "p90": round(percentile(latency_ms, 90), 3) if latency_ms else None,
            "p95": round(percentile(latency_ms, 95), 3) if latency_ms else None,
            "p99": round(percentile(latency_ms, 99), 3) if latency_ms else None,
            "max": round(max(latency_ms), 3) if latency_ms else None,
            "histogram": histogram(latency_ms, [4, 5, 6, 7, 8, 10, 12, 16, 24]) if latency_ms else {},
        },
        "step_latency_ms": {name: latency_summary(values) for name, values in step_latency_ms.items()},
        "detections_by_label": dict(sorted(detections_by_label.items())),
        "frames_by_detection_count": dict(sorted(frames_by_detection_count.items())),
        "nvidia": nvidia.summary(),
        "cpu": cpu.summary(),
        "rapl": {
            "available": bool(rapl_start and rapl_end),
            "average_w": rapl_delta_w(rapl_start, rapl_end, elapsed),
        },
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
