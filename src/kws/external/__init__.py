"""External KWS model integrations."""

from kws.external.hf_kws import (
    DEFAULT_EXTERNAL_AUX_MODEL_ID,
    DEFAULT_EXTERNAL_VERIFIER_MODEL_ID,
    ENSEMBLE_AST_SUPERB_MODEL_ID,
    SUPPORTED_EXTERNAL_MODEL_IDS,
    collect_external_probs_from_loader,
    ExternalKWSBatchResult,
    ExternalKWSLogitCache,
    benchmark_external_latency_ms,
    fit_external_verifier_calibration,
    predict_kws12_from_paths,
    predict_kws12_from_waveforms,
    slugify_model_id,
)

__all__ = [
    "DEFAULT_EXTERNAL_AUX_MODEL_ID",
    "DEFAULT_EXTERNAL_VERIFIER_MODEL_ID",
    "ENSEMBLE_AST_SUPERB_MODEL_ID",
    "SUPPORTED_EXTERNAL_MODEL_IDS",
    "collect_external_probs_from_loader",
    "ExternalKWSBatchResult",
    "ExternalKWSLogitCache",
    "benchmark_external_latency_ms",
    "fit_external_verifier_calibration",
    "predict_kws12_from_paths",
    "predict_kws12_from_waveforms",
    "slugify_model_id",
]
