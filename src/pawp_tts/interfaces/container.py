"""Dependency injection composition root wiring domain ports to adapters."""

from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv

from pawp_tts.application.audit_service import (
    CountModelParamsUseCase,
    EvaluateAsrUseCase,
    InspectCheckpointUseCase,
)
from pawp_tts.application.dataset_service import BuildDatasetManifestsUseCase
from pawp_tts.application.frontend_service import PhonemizeUtteranceUseCase
from pawp_tts.infrastructure.data.manifest_builder import BilingualManifestBuilder
from pawp_tts.infrastructure.evaluation.http_asr_adapter import HttpMultipartAsrAdapter
from pawp_tts.infrastructure.frontend.english_adapter import EnglishEspeakG2PAdapter
from pawp_tts.infrastructure.frontend.nepali_adapter import NepaliRuleG2PAdapter
from pawp_tts.infrastructure.persistence.ledger import JsonlExperimentLedger
from pawp_tts.infrastructure.weight_programming.checkpoint_inspector import (
    SafetensorsCheckpointInspector,
)
from pawp_tts.infrastructure.weight_programming.param_auditor import TorchParamAuditor


@dataclass(frozen=True)
class AppContainer:
    """Wired application use-case container."""

    phonemize: PhonemizeUtteranceUseCase
    count_params: CountModelParamsUseCase
    inspect_checkpoint: InspectCheckpointUseCase
    evaluate_asr: EvaluateAsrUseCase
    build_manifests: BuildDatasetManifestsUseCase


def build_container() -> AppContainer:
    """Load environment configuration and wire production adapters."""
    load_dotenv()

    en_g2p = EnglishEspeakG2PAdapter()
    ne_g2p = NepaliRuleG2PAdapter()
    ledger = JsonlExperimentLedger()

    return AppContainer(
        phonemize=PhonemizeUtteranceUseCase(english_g2p=en_g2p, nepali_g2p=ne_g2p),
        count_params=CountModelParamsUseCase(auditor=TorchParamAuditor(), ledger=ledger),
        inspect_checkpoint=InspectCheckpointUseCase(inspector=SafetensorsCheckpointInspector()),
        evaluate_asr=EvaluateAsrUseCase(asr=HttpMultipartAsrAdapter()),
        build_manifests=BuildDatasetManifestsUseCase(
            builder=BilingualManifestBuilder(),
            ledger=ledger,
        ),
    )
