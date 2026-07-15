"""Build validated per-script translation bundles for server and static modes."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dialogue_loader import DialogueLoader  # noqa: E402
from translation_cache import canonical_source_hash  # noqa: E402


TAG_RE = re.compile(r"\[[^\[\]]+\]")
SCRIPT_TOKEN_REPLACEMENTS = (
    ("[%1]", "\u85e4\u4e38\u7acb\u9999"),
    ("[line 3]", "\u2014\u2014"),
    ("[line 6]", "\u2014\u2014"),
    ("[line 18]", "\u2014\u2014"),
)


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _normalize_script_tokens(value: str) -> str:
    normalized = str(value or "")
    for token, replacement in SCRIPT_TOKEN_REPLACEMENTS:
        normalized = normalized.replace(token, replacement)
    return normalized


def _normalize_source_dialogues(dialogues: List[Dict]) -> List[Dict]:
    return [
        {
            "speaker": _normalize_script_tokens(item.get("speaker", "")),
            "content": _normalize_script_tokens(item.get("content", "")),
        }
        for item in dialogues
    ]


def _normalize_translations(translations: List[Dict]) -> List[Dict]:
    return [
        {
            "speaker": _normalize_script_tokens(item.get("speaker", "")),
            "translated_content": _normalize_script_tokens(
                item.get("translated_content", "")
            ),
        }
        for item in translations
    ]


def _war_script_ids(loader: DialogueLoader, war_id: str, region: str = "JP") -> List[str]:
    endpoint = f"{loader.db_loader.BASE_URL}/nice/{region}/war/{war_id}"
    war = loader.db_loader._make_request_with_retry(endpoint)
    ids = []
    for spot in war.get("spots", []) or []:
        for quest in spot.get("quests", []) or []:
            for phase in quest.get("phaseScripts", []) or []:
                for script in phase.get("scripts", []) or []:
                    script_id = str(script.get("scriptId", "")).strip()
                    if script_id and script_id != "0":
                        ids.append(script_id)
    return list(dict.fromkeys(ids))


def _bundle_payload(
    script_id: str,
    source_dialogues: List[Dict],
    translations: List[Dict],
    provider: str,
    model: str,
) -> Dict:
    return {
        "schema_version": 1,
        "script_id": script_id,
        "source_region": "JP",
        "source_hash": canonical_source_hash(source_dialogues),
        "target_language": "zh-CN",
        "provider": provider,
        "model": model,
        "prompt_version": "fgo-agent-v1",
        "dialogue_count": len(source_dialogues),
        "trusted_generation": True,
        "generator": {
            "app": "fgo_translator",
            "branch_mode": "bundled",
            "generated_at": "2026-07-15T00:00:00Z",
        },
        "translations": [
            {
                "speaker": str(item.get("speaker", "")),
                "translated_content": str(item.get("translated_content", "")),
            }
            for item in translations
        ],
    }


def _validate_agent_script(source: Dict, translated: Dict) -> None:
    script_id = source["script_id"]
    if translated.get("script_id") != script_id:
        raise ValueError(f"Script ID mismatch in {script_id}")
    if translated.get("source_hash") != source.get("source_hash"):
        raise ValueError(f"Source hash mismatch in {script_id}")
    source_lines = source.get("dialogues", [])
    translated_lines = translated.get("translations", [])
    if len(source_lines) != len(translated_lines):
        raise ValueError(
            f"Dialogue count mismatch in {script_id}: "
            f"{len(translated_lines)} != {len(source_lines)}"
        )
    for index, (original, result) in enumerate(zip(source_lines, translated_lines)):
        if not str(result.get("translated_content", "")).strip():
            raise ValueError(f"Empty translation in {script_id} line {index}")
        source_speaker_tags = TAG_RE.findall(str(original.get("speaker", "")))
        translated_speaker_tags = TAG_RE.findall(str(result.get("speaker", "")))
        if source_speaker_tags != translated_speaker_tags:
            raise ValueError(
                f"Speaker tag mismatch in {script_id} line {index}: "
                f"{source_speaker_tags!r} != {translated_speaker_tags!r}"
            )
        source_tags = TAG_RE.findall(str(original.get("content", "")))
        translated_tags = TAG_RE.findall(str(result.get("translated_content", "")))
        if source_tags != translated_tags:
            raise ValueError(
                f"Formatting tag mismatch in {script_id} line {index}: "
                f"{source_tags!r} != {translated_tags!r}"
            )


def build_agent_bundles(
    source_dir: Path,
    translated_dir: Path,
    output_dir: Path,
) -> List[str]:
    built = []
    source_paths = sorted(source_dir.glob("batch-*.json"))
    if not source_paths:
        raise ValueError(f"No source batches found in {source_dir}")
    for source_path in source_paths:
        translated_path = translated_dir / source_path.name
        if not translated_path.is_file():
            raise ValueError(f"Missing translated batch: {translated_path}")
        source_batch = _read_json(source_path)
        translated_batch = _read_json(translated_path)
        translated_by_id = {
            str(item.get("script_id")): item
            for item in translated_batch.get("scripts", [])
        }
        for source in source_batch.get("scripts", []):
            script_id = str(source["script_id"])
            translated = translated_by_id.get(script_id)
            if not translated:
                raise ValueError(f"Missing translated script {script_id} in {translated_path}")
            _validate_agent_script(source, translated)
            payload = _bundle_payload(
                script_id,
                _normalize_source_dialogues(source.get("dialogues", [])),
                _normalize_translations(translated.get("translations", [])),
                provider="codex-agent",
                model="agent-translation",
            )
            _write_json(output_dir / f"{script_id}.json", payload)
            built.append(script_id)
    return built


def build_official_bundles(
    loader: DialogueLoader,
    war_ids: Iterable[str],
    output_dir: Path,
) -> List[str]:
    built = []
    for war_id in war_ids:
        for script_id in _war_script_ids(loader, str(war_id), region="JP"):
            output_path = output_dir / f"{script_id}.json"
            if output_path.is_file():
                existing = _read_json(output_path)
                if existing.get("provider") == "codex-agent":
                    continue
            source = loader.extract_dialogues(script_id, region="JP")
            official = loader.extract_dialogues(script_id, region="CN")
            if len(source) != len(official):
                raise ValueError(
                    f"Atlas CN alignment mismatch in {script_id}: "
                    f"{len(official)} != {len(source)}"
                )
            translations = [
                {
                    "speaker": item.get("speaker", ""),
                    "translated_content": item.get("translated_content", item.get("content", "")),
                }
                for item in official
            ]
            payload = _bundle_payload(
                script_id,
                source,
                translations,
                provider="atlas-cn",
                model="official-sync",
            )
            _write_json(output_path, payload)
            built.append(script_id)
    return list(dict.fromkeys(built))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--translated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("translations/zh-CN"))
    parser.add_argument("--official-war", action="append", default=[])
    parser.add_argument("--static-output-dir", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    loader = DialogueLoader()
    agent_ids = build_agent_bundles(args.source_dir, args.translated_dir, args.output_dir)
    official_ids = build_official_bundles(loader, args.official_war, args.output_dir)
    index = {
        "schema_version": 1,
        "target_language": "zh-CN",
        "scripts": sorted(set(agent_ids + official_ids)),
        "agent_scripts": sorted(set(agent_ids)),
        "official_scripts": sorted(set(official_ids)),
    }
    _write_json(args.output_dir / "index.json", index)
    if args.static_output_dir:
        args.static_output_dir.mkdir(parents=True, exist_ok=True)
        for path in args.output_dir.glob("*.json"):
            shutil.copy2(path, args.static_output_dir / path.name)
    print(
        f"Built {len(agent_ids)} agent and {len(official_ids)} official bundles "
        f"in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
