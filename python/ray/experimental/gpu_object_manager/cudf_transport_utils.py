"""
Utilities for transporting cuDF DataFrames through Ray Direct Transport (RDT).

cuDF DataFrames are serialized by extracting GPU column buffers as
``torch.ByteTensor`` objects that travel via the existing NIXL/CUDA-IPC
transport, while the header (column names, dtypes, null-mask structure)
and any CPU-resident frames (string offsets, etc.) are inlined as
base64-encoded bytes in the msgpack payload.

No CPU round-trip is performed for GPU-resident column data: the only
CUDA operation on the send side is one GPU-to-GPU ``memcpy`` per column
buffer to create an *owned* copy, so the original DataFrame can be
garbage-collected safely while the byte tensors are held in the
``GPUObjectStore``.
"""

import base64
import logging
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import cudf

logger = logging.getLogger(__name__)

# Marker embedded in the serialized payload to identify cuDF DataFrames.
_CUDF_DF_TYPE_MARKER = "__cudf_df__"


def register_cudf_serializer() -> None:
    """Register a cloudpickle custom serializer/deserializer for ``cudf.DataFrame``.

    Must be called on every worker process that will send or receive a
    ``cudf.DataFrame`` through RDT. The function is idempotent — calling it
    multiple times on the same process has no additional effect beyond the first
    registration (the second call simply overwrites the dispatch entry with an
    equivalent function).

    This mirrors the pattern used by
    ``TorchTensorType.register_custom_serializer()`` in
    ``torch_tensor_type.py``.
    """
    import cudf

    import ray.util.serialization

    ray.util.serialization.register_serializer(
        cudf.DataFrame,
        serializer=_serialize_cudf_df,
        deserializer=_deserialize_cudf_df,
    )


def _serialize_cudf_df(df: "cudf.DataFrame") -> Dict[str, Any]:
    """Cloudpickle serializer for ``cudf.DataFrame``.

    Called automatically by cloudpickle whenever a ``cudf.DataFrame`` is
    encountered during ``serialize_gpu_objects``.  Each GPU-resident column
    buffer is copied into an owned ``torch.ByteTensor`` and appended to
    ``_SerializationContext._out_of_band_tensors``; its position in that list
    becomes an integer placeholder stored in the returned payload dict.
    CPU-resident frames (e.g. string offsets) are base64-encoded inline.

    Args:
        df: The ``cudf.DataFrame`` to serialize.

    Returns:
        A payload dict containing the cuDF header and frame descriptors.

    Raises:
        RuntimeError: If called outside of an RDT serialization context
            (i.e. ``_use_external_transport`` is False).
    """
    import cupy as cp
    import torch

    from ray.experimental.channel import ChannelContext

    ctx = ChannelContext.get_current().serialization_context

    if not ctx.use_external_transport:
        raise RuntimeError(
            "cudf.DataFrame must be returned from a method decorated with "
            "@ray.method(tensor_transport=...). Direct object-store transport "
            "for cudf.DataFrame is not supported."
        )

    header, frames = df.serialize()

    frame_descs: List[Dict[str, Any]] = []
    for frame in frames:
        if hasattr(frame, "__cuda_array_interface__"):
            # GPU-resident frame: copy into an owned torch.ByteTensor so the
            # original DataFrame can be GC'd without dangling CUDA pointers.
            cupy_view = cp.asarray(frame).view(cp.uint8)
            n = len(cupy_view)
            device = f"cuda:{cupy_view.device.id}"
            owned = torch.empty(n, dtype=torch.uint8, device=device)
            # GPU-to-GPU copy — no CPU round-trip.  torch.as_tensor on a cupy
            # array creates a zero-copy tensor view via __cuda_array_interface__;
            # owned.copy_() then performs a device-to-device memcpy.
            owned.copy_(torch.as_tensor(cupy_view))

            idx = len(ctx._out_of_band_tensors)
            ctx._out_of_band_tensors.append(owned)
            frame_descs.append({"kind": "gpu", "tensor_idx": idx})
        else:
            # CPU-resident frame (bytes, numpy array, or cuDF HostBuffer).
            try:
                raw = bytes(frame)
            except TypeError:
                # Fallback for objects that need explicit conversion.
                import numpy as np

                raw = np.asarray(frame).tobytes()
            frame_descs.append(
                {
                    "kind": "cpu",
                    "data": base64.b64encode(raw).decode("ascii"),
                }
            )

    return {
        "__type__": _CUDF_DF_TYPE_MARKER,
        "header": header,
        "frame_descs": frame_descs,
    }


def _deserialize_cudf_df(payload: Dict[str, Any]) -> "cudf.DataFrame":
    """Cloudpickle deserializer for ``cudf.DataFrame``.

    Reconstructs the ``cudf.DataFrame`` from the payload dict produced by
    :func:`_serialize_cudf_df`.  GPU frames are retrieved from
    ``_SerializationContext._out_of_band_tensors`` by their placeholder index
    and converted to cupy arrays; CPU frames are base64-decoded back to bytes.

    Args:
        payload: The dict returned by :func:`_serialize_cudf_df`.

    Returns:
        The reconstructed ``cudf.DataFrame``.
    """
    import cudf
    import cupy as cp

    from ray.experimental.channel import ChannelContext

    ctx = ChannelContext.get_current().serialization_context

    frames = []
    for desc in payload["frame_descs"]:
        if desc["kind"] == "gpu":
            tensor = ctx._out_of_band_tensors[desc["tensor_idx"]]
            # Convert the ByteTensor back to a cupy uint8 array.  cuDF's
            # deserialize() accepts any object implementing
            # __cuda_array_interface__.
            cupy_arr = cp.asarray(tensor)
            frames.append(cupy_arr)
        else:
            frames.append(base64.b64decode(desc["data"]))

    return cudf.DataFrame.deserialize(payload["header"], frames)
