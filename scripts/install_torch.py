import json
import os
import re
import subprocess
import sys


TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]
TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"
TORCHAUDIO_VERSION = "2.6.0"
OFFICIAL_CU128_TORCH_VERSION = "2.8.0"
OFFICIAL_CU128_TORCHVISION_VERSION = "0.23.0"
OFFICIAL_CU128_TORCHAUDIO_VERSION = "2.8.0"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CUDA_CANDIDATES = [
    ("cu128", "https://download.pytorch.org/whl/cu128", False, 128),
    ("cu124", "https://download.pytorch.org/whl/cu124", False, 124),
    ("cu121", "https://download.pytorch.org/whl/cu121", False, 121),
    ("cu118", "https://download.pytorch.org/whl/cu118", False, 118),
]
CPU_CANDIDATE = ("cpu", CPU_INDEX, False, None)
LOG_FILE = os.getenv("CHATTERBOX_INSTALL_LOG")


def log(message):
    print(message)
    if not LOG_FILE:
        return
    with open(LOG_FILE, "a", encoding="utf-8") as log_handle:
        log_handle.write(message + "\n")


def log_block(prefix, payload):
    if not payload:
        return
    for line in payload.splitlines():
        log(f"{prefix}{line}")


def _parse_version_token(version_str):
    parts = version_str.strip().split(".")
    if len(parts) < 2:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except ValueError:
        return None
    return major, minor


def _version_to_tag(version_str):
    parsed = _parse_version_token(version_str)
    if parsed is None:
        return None
    major, minor = parsed
    return major * 10 + minor


def detect_nvidia_cuda_version():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=cuda_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        first_line = next(
            (line.strip() for line in output.splitlines() if line.strip()),
            None,
        )
        if first_line:
            return _version_to_tag(first_line), "nvidia-smi"
    except (FileNotFoundError, subprocess.CalledProcessError, StopIteration):
        pass

    # Newer drivers reject the cuda_version query field and label the header
    # "CUDA UMD Version" instead of "CUDA Version"; parse the plain header.
    try:
        output = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        match = re.search(r"CUDA (?:UMD )?Version:\s*([0-9]+\.[0-9]+)", output)
        if match:
            return _version_to_tag(match.group(1)), "nvidia-smi header"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        output = subprocess.check_output(
            ["nvcc", "--version"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            if "release" in line:
                version_str = line.split("release")[-1].strip().split(",")[0]
                parsed = _version_to_tag(version_str)
                if parsed is not None:
                    return parsed, "nvcc"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return None, None


def choose_install_candidates(cuda_version):
    if cuda_version is None:
        return [CPU_CANDIDATE]

    # Newer NVIDIA generations need a matching modern wheel. Falling back to
    # older CUDA runtime builds only wastes time because those builds do not
    # contain kernels for the newer GPU architecture.
    if cuda_version >= 128:
        return [CUDA_CANDIDATES[0], CPU_CANDIDATE]

    candidates = [
        candidate
        for candidate in CUDA_CANDIDATES
        if cuda_version >= candidate[3]
    ]
    if not candidates:
        return [CPU_CANDIDATE]
    candidates.append(CPU_CANDIDATE)
    return candidates


def get_package_versions(label):
    if label == "cu128":
        return (
            OFFICIAL_CU128_TORCH_VERSION,
            OFFICIAL_CU128_TORCHVISION_VERSION,
            OFFICIAL_CU128_TORCHAUDIO_VERSION,
        )
    return TORCH_VERSION, TORCHVISION_VERSION, TORCHAUDIO_VERSION


def install_torch(label, index_url, prerelease):
    log(f"Installing from: {index_url}")
    torch_version, torchvision_version, torchaudio_version = get_package_versions(label)
    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--upgrade",
        "--reinstall",
        "--strict",
        "--no-deps",
        f"torch=={torch_version}",
        f"torchvision=={torchvision_version}",
        f"torchaudio=={torchaudio_version}",
        "--index-url",
        index_url,
    ]
    if prerelease:
        cmd.extend(["--prerelease", "allow"])

    log(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    log_block("[uv stdout] ", result.stdout)
    log_block("[uv stderr] ", result.stderr)


def verify_torch(expect_cuda):
    verify_code = """
import json
import sys

try:
    import torch
except Exception as exc:
    print(json.dumps({"ok": False, "error": repr(exc)}))
    raise SystemExit(0)

payload = {
    "ok": True,
    "torch_version": getattr(torch, "__version__", "unknown"),
    "cuda_available": bool(torch.cuda.is_available()),
    "torch_cuda_version": getattr(torch.version, "cuda", None),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
    "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "cuda_runtime_ok": False,
    "cuda_runtime_error": None,
}

if payload["cuda_available"]:
    try:
        x = torch.randn((8, 8), device="cuda")
        y = x @ x
        _ = y.sum().item()
        gru = torch.nn.GRU(4, 4).to("cuda")
        gru(torch.randn((2, 1, 4), device="cuda"))
        payload["cuda_runtime_ok"] = True
    except Exception as exc:
        payload["cuda_runtime_error"] = repr(exc)

print(json.dumps(payload))
"""
    result = subprocess.run(
        [sys.executable, "-c", verify_code],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1]) if lines else {"ok": False, "error": "No verification output"}
    if not payload.get("ok"):
        return False, payload
    if expect_cuda and not payload.get("cuda_available"):
        return False, payload
    if expect_cuda and not payload.get("cuda_runtime_ok"):
        return False, payload
    return True, payload


def main():
    cuda_version, detection_source = detect_nvidia_cuda_version()
    if cuda_version is None:
        log("No NVIDIA CUDA runtime detected. Installing CPU PyTorch build.")
    else:
        log(f"Detected NVIDIA CUDA capability {cuda_version} via {detection_source}.")

    candidates = choose_install_candidates(cuda_version)
    log(f"Install candidates: {candidates}")
    failures = []

    for label, index_url, prerelease, minimum_cuda in candidates:
        expect_cuda = label != "cpu"
        log(f"Trying PyTorch target: {label}")
        try:
            install_torch(label, index_url, prerelease)
            ok, payload = verify_torch(expect_cuda=expect_cuda)
            log(f"Verification payload for {label}: {json.dumps(payload, ensure_ascii=False)}")
            if ok:
                mode = "CUDA" if payload.get("cuda_available") else "CPU"
                log(
                    f"PyTorch verification succeeded. Mode={mode}, "
                    f"torch={payload.get('torch_version')}, "
                    f"torch_cuda={payload.get('torch_cuda_version')}, "
                    f"devices={payload.get('device_count')}"
                )
                return 0

            reason = (
                f"verification failed: cuda_available={payload.get('cuda_available')} "
                f"torch_cuda={payload.get('torch_cuda_version')} "
                f"arch_list={payload.get('arch_list')} "
                f"device_capability={payload.get('device_capability')} "
                f"runtime_error={payload.get('cuda_runtime_error')} "
                f"error={payload.get('error')}"
            )
            log(f"PyTorch target {label} did not verify correctly: {reason}")
            failures.append((label, reason))
        except subprocess.CalledProcessError as exc:
            reason = f"installer exited with code {exc.returncode}"
            log_block("[uv stdout] ", exc.stdout)
            log_block("[uv stderr] ", exc.stderr)
            log(f"PyTorch target {label} failed: {reason}")
            failures.append((label, reason))

        if minimum_cuda is not None and not (
            cuda_version is not None and cuda_version >= 128
        ):
            log("Falling back to the next compatible PyTorch runtime.")

    log("Unable to install a working PyTorch configuration.")
    if failures:
        log("Attempts:")
        for label, reason in failures:
            log(f"  - {label}: {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
