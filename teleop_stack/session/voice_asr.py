from __future__ import annotations

import json
import wave
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from teleop_stack.session.voice_controls import VoiceTeleopCommand


@dataclass(frozen=True)
class VoiceCommandPhraseConfig:
    engage_aliases: tuple[str, ...] = (
        "engage",
        "start",
        "start follow",
        "start teleop",
        "开始",
        "接管",
    )
    clutch_aliases: tuple[str, ...] = (
        "clutch",
        "hold",
        "pause",
        "暂停",
        "保持",
    )
    resume_aliases: tuple[str, ...] = (
        "resume",
        "continue",
        "继续",
        "恢复",
    )
    recenter_aliases: tuple[str, ...] = (
        "recenter",
        "reset",
        "realign",
        "重置",
    )
    stop_aliases: tuple[str, ...] = (
        "stop",
        "停止",
        "停止遥操",
        "结束遥操",
    )
    exit_aliases: tuple[str, ...] = (
        "exit",
        "park",
        "退出",
        "退出遥操",
        "收回",
        "回桌面",
    )
    estop_aliases: tuple[str, ...] = (
        "estop",
        "e stop",
        "emergency stop",
        "急停",
    )
    success_aliases: tuple[str, ...] = (
        "success",
        "succeed",
        "succeeded",
        "task success",
        "成功",
        "任务成功",
    )
    failure_aliases: tuple[str, ...] = (
        "failure",
        "fail",
        "failed",
        "task failure",
        "失败",
        "任务失败",
    )
    abort_aliases: tuple[str, ...] = (
        "abort",
        "give up",
        "operator abort",
        "放弃",
        "结束任务",
        "终止任务",
    )


def _normalize_phrase(text: str) -> str:
    normalized = " ".join(str(text).strip().lower().replace("_", " ").split())
    return normalized


def _compact_phrase(text: str) -> str:
    return "".join(_normalize_phrase(text).split())


_COMMAND_EDGE_PUNCTUATION = " \t\r\n,.;:!?，。；：！？、\"'“”‘’（）()[]{}<>《》"


def _normalize_transcript_for_command(text: str) -> str:
    return _normalize_phrase(str(text).strip(_COMMAND_EDGE_PUNCTUATION))


def _candidate_phrase_forms(text: str) -> tuple[str, ...]:
    normalized = _normalize_phrase(text)
    compact = _compact_phrase(text)
    forms: list[str] = []
    if normalized:
        forms.append(normalized)
    if compact and compact != normalized:
        forms.append(compact)
    return tuple(forms)


def normalize_transcript_to_command(
    transcript: str,
    *,
    phrase_config: VoiceCommandPhraseConfig | None = None,
) -> VoiceTeleopCommand | None:
    config = phrase_config or VoiceCommandPhraseConfig()
    text = _normalize_transcript_for_command(transcript)
    if not text:
        return None
    text_forms = _candidate_phrase_forms(text)

    ordered_aliases: tuple[tuple[VoiceTeleopCommand, tuple[str, ...]], ...] = (
        ("recenter", config.recenter_aliases),
        ("resume", config.resume_aliases),
        ("clutch", config.clutch_aliases),
        ("engage", config.engage_aliases),
        ("success", config.success_aliases),
        ("failure", config.failure_aliases),
        ("exit", config.exit_aliases),
        ("abort", config.abort_aliases),
        ("estop", config.estop_aliases),
        ("stop", config.stop_aliases),
    )
    for command, aliases in ordered_aliases:
        for alias in aliases:
            alias_forms = _candidate_phrase_forms(alias)
            if any(form == candidate for form in alias_forms for candidate in text_forms):
                return command
    return None


@dataclass(frozen=True)
class RecognizedVoiceCommand:
    transcript: str
    command: VoiceTeleopCommand | None
    confidence: float | None = None


def confidence_meets_threshold(confidence: float | None, min_confidence: float) -> bool:
    threshold = float(min_confidence)
    if threshold <= 0.0:
        return True
    if confidence is None:
        return False
    return float(confidence) >= threshold


@dataclass(frozen=True)
class VoskAsrConfig:
    model_path: Path
    sample_rate_hz: int = 16000
    phrase_config: VoiceCommandPhraseConfig = field(default_factory=VoiceCommandPhraseConfig)
    grammar_constrained: bool = False


class VoskVoiceCommandRecognizer:
    def __init__(self, config: VoskAsrConfig):
        self.config = config
        self._vosk = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import vosk
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "vosk is not installed. Install the optional ASR dependencies with "
                '`python -m pip install -e ".[voice_asr]"` first.'
            ) from exc
        self._vosk = vosk
        self._model = vosk.Model(str(self.config.model_path))

    def _grammar_phrases(self) -> list[str]:
        phrase_config = self.config.phrase_config
        all_aliases = (
            phrase_config.engage_aliases
            + phrase_config.clutch_aliases
            + phrase_config.resume_aliases
            + phrase_config.recenter_aliases
            + phrase_config.stop_aliases
            + phrase_config.exit_aliases
            + phrase_config.estop_aliases
            + phrase_config.success_aliases
            + phrase_config.failure_aliases
            + phrase_config.abort_aliases
        )
        deduped: list[str] = []
        seen: set[str] = set()
        for alias in all_aliases:
            normalized = _normalize_phrase(alias)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        return deduped

    def _build_recognizer(self):
        self._ensure_model()
        assert self._vosk is not None
        assert self._model is not None

        if self.config.grammar_constrained:
            grammar = json.dumps(self._grammar_phrases(), ensure_ascii=False)
            recognizer = self._vosk.KaldiRecognizer(
                self._model,
                float(self.config.sample_rate_hz),
                grammar,
            )
        else:
            recognizer = self._vosk.KaldiRecognizer(self._model, float(self.config.sample_rate_hz))
        recognizer.SetWords(True)
        return recognizer

    def _result_from_raw(self, raw_result: dict[str, object]) -> RecognizedVoiceCommand:
        transcript = str(raw_result.get("text", "")).strip()
        confidence = _mean_confidence(raw_result.get("result"))
        command = normalize_transcript_to_command(transcript, phrase_config=self.config.phrase_config)
        return RecognizedVoiceCommand(
            transcript=transcript,
            command=command,
            confidence=confidence,
        )

    def transcribe_wav(self, wav_path: Path) -> RecognizedVoiceCommand:
        with wave.open(str(wav_path), "rb") as handle:
            if handle.getnchannels() != 1:
                raise ValueError(f"Expected mono wav, got {handle.getnchannels()} channels.")
            if handle.getsampwidth() != 2:
                raise ValueError(f"Expected 16-bit wav, got sampwidth={handle.getsampwidth()}.")
            if handle.getframerate() != self.config.sample_rate_hz:
                raise ValueError(f"Expected sample_rate={self.config.sample_rate_hz}, got {handle.getframerate()}.")

            recognizer = self._build_recognizer()
            while True:
                chunk = handle.readframes(4000)
                if not chunk:
                    break
                recognizer.AcceptWaveform(chunk)

            raw_result = json.loads(recognizer.FinalResult())

        return self._result_from_raw(raw_result)


class StreamingVoskVoiceCommandRecognizer:
    def __init__(self, config: VoskAsrConfig):
        self._base = VoskVoiceCommandRecognizer(config)
        self._recognizer = self._base._build_recognizer()

    @property
    def config(self) -> VoskAsrConfig:
        return self._base.config

    def accept_pcm16le(self, data: bytes) -> RecognizedVoiceCommand | None:
        if not data:
            return None
        accepted = self._recognizer.AcceptWaveform(data)
        if not accepted:
            return None
        raw_result = json.loads(self._recognizer.Result())
        return self._base._result_from_raw(raw_result)

    def finalize(self) -> RecognizedVoiceCommand | None:
        raw_result = json.loads(self._recognizer.FinalResult())
        result = self._base._result_from_raw(raw_result)
        if not result.transcript:
            return None
        return result


def _extract_funasr_text(result: object) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        value = result.get("text")
        if value is not None:
            return str(value).strip()
        return ""
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            text = _extract_funasr_text(item)
            if text:
                parts.append(text)
        return "".join(parts).strip()
    return ""


def _extract_funasr_vad_events(result: object) -> tuple[tuple[int, int], ...]:
    if result is None:
        return ()
    if isinstance(result, dict):
        value = result.get("value")
        if not isinstance(value, list):
            return ()
        events: list[tuple[int, int]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                events.append((int(item[0]), int(item[1])))
            except Exception:
                continue
        return tuple(events)
    if isinstance(result, list):
        events: list[tuple[int, int]] = []
        for item in result:
            events.extend(_extract_funasr_vad_events(item))
        return tuple(events)
    return ()


@dataclass(frozen=True)
class FunasrStreamingConfig:
    model: str = "paraformer-zh-streaming"
    vad_model: str = "fsmn-vad"
    device: str = "cpu"
    sample_rate_hz: int = 16000
    chunk_size: tuple[int, int, int] = (0, 10, 5)
    encoder_chunk_look_back: int = 4
    decoder_chunk_look_back: int = 1
    vad_chunk_size_ms: int = 200
    phrase_config: VoiceCommandPhraseConfig = field(default_factory=VoiceCommandPhraseConfig)

    @property
    def asr_chunk_samples(self) -> int:
        # FunASR streaming examples use chunk_size=[0, 10, 5] and
        # chunk_stride=chunk_size[1] * 960 at 16 kHz, i.e. 600 ms chunks.
        return max(int(self.chunk_size[1]) * 960, 1)

    @property
    def vad_chunk_samples(self) -> int:
        return max(int(self.sample_rate_hz * self.vad_chunk_size_ms / 1000), 1)


class StreamingFunasrVoiceCommandRecognizer:
    """FunASR streaming recognizer for short command phrases.

    The ASR model runs on fixed paraformer streaming chunks. FSMN-VAD is used
    to decide when to flush the current utterance. Command forwarding remains a
    strict whitelist decision through normalize_transcript_to_command().
    """

    def __init__(self, config: FunasrStreamingConfig):
        self.config = config
        try:
            import numpy as np
            from funasr import AutoModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "funasr backend is not installed. Install it with "
                '`python -m pip install -e ".[voice_asr_funasr]"` in the active env.'
            ) from exc

        self._np = np
        self._asr_model = AutoModel(model=config.model, device=config.device)
        self._vad_model = AutoModel(model=config.vad_model, device=config.device)
        self._asr_cache: dict = {}
        self._vad_cache: dict = {}
        self._asr_buffer = np.empty(0, dtype=np.float32)
        self._vad_buffer = np.empty(0, dtype=np.float32)
        self._last_text = ""

    def accept_pcm16le(self, data: bytes) -> RecognizedVoiceCommand | None:
        if not data:
            return None
        samples = self._pcm16le_to_float32(data)
        if samples.size == 0:
            return None

        self._asr_buffer = self._np.concatenate([self._asr_buffer, samples])
        self._vad_buffer = self._np.concatenate([self._vad_buffer, samples])

        result: RecognizedVoiceCommand | None = None
        while self._asr_buffer.size >= self.config.asr_chunk_samples:
            chunk = self._asr_buffer[: self.config.asr_chunk_samples]
            self._asr_buffer = self._asr_buffer[self.config.asr_chunk_samples :]
            result = self._feed_asr_chunk(chunk, is_final=False) or result

        should_finalize = False
        while self._vad_buffer.size >= self.config.vad_chunk_samples:
            chunk = self._vad_buffer[: self.config.vad_chunk_samples]
            self._vad_buffer = self._vad_buffer[self.config.vad_chunk_samples :]
            should_finalize = self._feed_vad_chunk(chunk) or should_finalize

        if should_finalize:
            result = self._finalize_utterance() or result
        return result

    def finalize(self) -> RecognizedVoiceCommand | None:
        return self._finalize_utterance()

    def _pcm16le_to_float32(self, data: bytes):
        usable_len = len(data) - (len(data) % 2)
        if usable_len <= 0:
            return self._np.empty(0, dtype=self._np.float32)
        int16 = self._np.frombuffer(data[:usable_len], dtype="<i2")
        return int16.astype(self._np.float32) / 32768.0

    def _feed_vad_chunk(self, chunk) -> bool:
        result = self._vad_model.generate(
            input=chunk,
            cache=self._vad_cache,
            is_final=False,
            chunk_size=int(self.config.vad_chunk_size_ms),
        )
        events = _extract_funasr_vad_events(result)
        return any(end_ms >= 0 for _, end_ms in events)

    def _feed_asr_chunk(self, chunk, *, is_final: bool) -> RecognizedVoiceCommand | None:
        result = self._asr_model.generate(
            input=chunk,
            cache=self._asr_cache,
            is_final=bool(is_final),
            chunk_size=list(self.config.chunk_size),
            encoder_chunk_look_back=int(self.config.encoder_chunk_look_back),
            decoder_chunk_look_back=int(self.config.decoder_chunk_look_back),
        )
        text = _extract_funasr_text(result)
        if text:
            transcript = self._merge_streaming_text(text)
            command = normalize_transcript_to_command(transcript, phrase_config=self.config.phrase_config)
            if command is not None:
                return RecognizedVoiceCommand(transcript=transcript, command=command, confidence=1.0)
        return None

    def _merge_streaming_text(self, text: str) -> str:
        text = str(text).strip()
        if not self._last_text:
            self._last_text = text
        elif text.startswith(self._last_text):
            self._last_text = text
        elif text not in self._last_text:
            self._last_text = f"{self._last_text}{text}"
        return self._last_text

    def _finalize_utterance(self) -> RecognizedVoiceCommand | None:
        if self._asr_buffer.size > 0:
            self._feed_asr_chunk(self._asr_buffer, is_final=True)

        transcript = self._last_text.strip()
        self._asr_cache = {}
        self._vad_cache = {}
        self._asr_buffer = self._np.empty(0, dtype=self._np.float32)
        self._vad_buffer = self._np.empty(0, dtype=self._np.float32)
        self._last_text = ""
        if not transcript:
            return None
        command = normalize_transcript_to_command(transcript, phrase_config=self.config.phrase_config)
        return RecognizedVoiceCommand(
            transcript=transcript,
            command=command,
            confidence=1.0 if command is not None else None,
        )


def _mean_confidence(raw_words: object) -> float | None:
    if not isinstance(raw_words, Iterable):
        return None
    confidences: list[float] = []
    for item in raw_words:
        if isinstance(item, dict) and "conf" in item:
            try:
                confidences.append(float(item["conf"]))
            except Exception:
                continue
    if not confidences:
        return None
    return sum(confidences) / len(confidences)
