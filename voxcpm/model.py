"""VoxCPM model wrapper and inference utilities."""

import os
import logging
from typing import Optional, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


class VoxCPMModel:
    """Wrapper class for the VoxCPM speech recognition model.

    Handles model loading, audio preprocessing, and inference.
    """

    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_MAX_LENGTH = 448

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the VoxCPM model.

        Args:
            model_dir: Path to the model directory containing weights and config.
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.).
                    Defaults to CUDA if available, otherwise CPU.
            dtype: Torch dtype for model weights. Defaults to float16 on CUDA,
                   float32 on CPU.
        """
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if "cuda" in self.device else torch.float32)

        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> "VoxCPMModel":
        """Load model weights and processor from model_dir.

        Returns:
            self, for chaining.

        Raises:
            FileNotFoundError: If model_dir does not exist.
            RuntimeError: If model loading fails.
        """
        if self._loaded:
            logger.debug("Model already loaded, skipping.")
            return self

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        logger.info("Loading VoxCPM model from %s on %s", self.model_dir, self.device)

        try:
            # Lazy imports to avoid hard dependency at module level
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            self.processor = AutoProcessor.from_pretrained(self.model_dir)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_dir,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info("Model loaded successfully.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load VoxCPM model: {exc}") from exc

        return self

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded."""
        return self._loaded

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_LENGTH,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Raw audio waveform as a numpy array (float32, mono) or a
                   path to a WAV file.
            sample_rate: Sample rate of the audio in Hz.
            language: BCP-47 language tag (e.g. 'zh', 'en'). When None the
                      model will auto-detect the language.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Transcribed text string.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if not self._loaded:
            raise RuntimeError(
                "Model is not loaded. Call VoxCPMModel.load() first."
            )

        # Accept file paths for convenience
        if isinstance(audio, str):
            audio = self._load_wav(audio)

        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        generate_kwargs = {"max_new_tokens": max_new_tokens}
        if language is not None:
            generate_kwargs["language"] = language

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **generate_kwargs)

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription[0].strip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_wav(path: str) -> np.ndarray:
        """Load a WAV file and return a float32 mono waveform."""
        import soundfile as sf

        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            # Downmix to mono
            waveform = waveform.mean(axis=1)
        return waveform
