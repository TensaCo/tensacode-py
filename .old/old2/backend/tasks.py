from typing import Protocol, TypeVar, runtime_checkable

import ivy

_T = TypeVar("_T")


@runtime_checkable
class SupportsTextCompletion(Protocol):
    def complete(
        self,
        input_text: str,
        temperature,
        max_tokens,
        top_k,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop,
        **kwargs,
    ) -> str:
        raise NotImplementedError


@runtime_checkable
class SupportsTextEditing(Protocol):
    def edit(
        self,
        input_text: str,
        prompt: str,
        temperature,
        max_tokens,
        top_k,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop,
        **kwargs,
    ) -> str:
        raise NotImplementedError


Image = ivy.Array


@runtime_checkable
class SupportsImageGeneration(Protocol):
    def generate_image(
        self,
        prompt: str,
        temperature,
        rounds,
        **kwargs,
    ) -> Image:
        raise NotImplementedError


@runtime_checkable
class SupportsImageEditing(Protocol):
    def edit_image(
        self,
        image: Image,
        prompt: str,
        temperature,
        rounds,
        **kwargs,
    ) -> Image:
        raise NotImplementedError


FrameSequence = ivy.Array


@runtime_checkable
class SupportsNextFramePrediction(Protocol):
    def predict_next_frame(
        self,
        previous_frames: FrameSequence,
        temperature,
        rounds,
        **kwargs,
    ) -> Image:
        raise NotImplementedError


@runtime_checkable
class SupportsVideoCompletion(Protocol):
    def predict_video(
        self,
        previous_frames: FrameSequence,
        temperature,
        max_frames,
        top_k,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop,
        rounds_per_frame,
        **kwargs,
    ) -> Image:
        raise NotImplementedError


Audio = ivy.Array


@runtime_checkable
def SupportsTextToSpeech(Protocol):
    def text_to_speech(
        self,
        text: str,
        **kwargs,
    ) -> Audio:
        raise NotImplementedError


@runtime_checkable
def SupportsSpeechToText(Protocol):
    def speech_to_text(
        self,
        audio: Audio,
        **kwargs,
    ) -> str:
        raise NotImplementedError


@runtime_checkable
def SupportsAudioCompletion(Protocol):
    def complete_audio(
        self,
        audio: Audio,
        prompt: str,
        temperature,
        max_tokens,
        top_k,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop,
        **kwargs,
    ) -> Audio:
        raise NotImplementedError


@runtime_checkable
def SupportsAudioGeneration(Protocol):
    def generate_audio(
        self,
        prompt: str,
        temperature,
        max_tokens,
        top_k,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop,
        **kwargs,
    ) -> Audio:
        raise NotImplementedError


@runtime_checkable
def SupportsAudioEditing(Protocol):
    def edit_audio(
        self,
        audio: Audio,
        prompt: str,
        temperature,
        max_tokens,
        top_k,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop,
        **kwargs,
    ) -> Audio:
        raise NotImplementedError
