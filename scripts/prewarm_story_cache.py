"""Prewarm the local Atlas raw story script cache without translating text."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dialogue_loader import DialogueLoader  # noqa: E402


DEFAULT_QUESTS = ("3000801",)
DEFAULT_REGIONS = ("JP", "NA", "CN", "TW")
DEFAULT_PHASES = (1, 2)


@dataclass(frozen=True)
class ScriptCacheResult:
    region: str
    script_id: str
    quest_id: str
    phase: int
    success: bool


def _ordered_unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _phase_script_ids(quest_data: Dict) -> List[str]:
    script_ids = []
    for script in quest_data.get("scripts", []) or []:
        script_id = str(script.get("scriptId", "")).strip()
        if script_id and script_id != "0":
            script_ids.append(script_id)
    return _ordered_unique(script_ids)


def discover_story_scripts(
    loader: DialogueLoader,
    quests: Sequence[str],
    regions: Sequence[str],
    phases: Sequence[int],
    output: Callable[[str], None] = print,
) -> Dict[Tuple[str, int, str], List[str]]:
    """Discover script IDs per quest phase and Atlas region."""
    discovered: Dict[Tuple[str, int, str], List[str]] = {}
    for quest_id in quests:
        for phase in phases:
            for region in regions:
                endpoint = (
                    f"{loader.db_loader.BASE_URL}/nice/{region}/quest/"
                    f"{quest_id}/{phase}"
                )
                try:
                    quest_data = loader.db_loader._make_request_with_retry(endpoint)
                    script_ids = _phase_script_ids(quest_data or {})
                    discovered[(quest_id, phase, region)] = script_ids
                    if not script_ids:
                        output(f"{region}/quest {quest_id}/phase {phase}: no scripts")
                except Exception as exc:
                    discovered[(quest_id, phase, region)] = []
                    output(f"{region}/quest {quest_id}/phase {phase}: unavailable ({exc})")
    return discovered


def prewarm_story_cache(
    loader: DialogueLoader,
    quests: Sequence[str] = DEFAULT_QUESTS,
    regions: Sequence[str] = DEFAULT_REGIONS,
    phases: Sequence[int] = DEFAULT_PHASES,
    refresh: bool = False,
    output: Callable[[str], None] = print,
) -> List[ScriptCacheResult]:
    """Fetch discovered Atlas scripts into the loader's local raw-text cache."""
    normalized_regions = _ordered_unique(loader.normalize_region(region) for region in regions)
    normalized_quests = _ordered_unique(str(quest).strip() for quest in quests)
    normalized_phases = list(dict.fromkeys(int(phase) for phase in phases))
    discovered = discover_story_scripts(
        loader,
        normalized_quests,
        normalized_regions,
        normalized_phases,
        output=output,
    )

    results = []
    seen_region_scripts = set()
    for (quest_id, phase, region), script_ids in discovered.items():
        for script_id in script_ids:
            cache_key = (region, script_id)
            if cache_key in seen_region_scripts:
                continue
            seen_region_scripts.add(cache_key)
            text = loader.load_script_text(script_id, region=region, refresh=refresh)
            success = bool(text)
            results.append(
                ScriptCacheResult(
                    region=region,
                    script_id=script_id,
                    quest_id=quest_id,
                    phase=phase,
                    success=success,
                )
            )
            status = "success" if success else "missing"
            output(f"{region}/{script_id}: {status} (quest {quest_id}, phase {phase})")

    counts = defaultdict(lambda: {"success": 0, "missing": 0})
    for result in results:
        key = "success" if result.success else "missing"
        counts[result.region][key] += 1

    output("Summary:")
    for region in normalized_regions:
        output(
            f"{region}: success={counts[region]['success']} "
            f"missing={counts[region]['missing']}"
        )
    output(
        f"TOTAL: success={sum(values['success'] for values in counts.values())} "
        f"missing={sum(values['missing'] for values in counts.values())}"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prewarm local Atlas raw story scripts; no translation APIs are called."
    )
    parser.add_argument(
        "--quest",
        dest="quests",
        action="append",
        help="Quest ID to prewarm (repeatable; default: 3000801).",
    )
    parser.add_argument(
        "--region",
        dest="regions",
        action="append",
        type=str.upper,
        choices=("JP", "NA", "CN", "TW", "KR"),
        help="Atlas region to prewarm (repeatable; default: JP, NA, CN, TW).",
    )
    parser.add_argument(
        "--phase",
        dest="phases",
        action="append",
        type=int,
        help="Quest phase to inspect (repeatable; default: 1 and 2).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Local cache directory (default: cache).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh existing entries, falling back to cached text on network failure.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loader = DialogueLoader(cache_dir=args.cache_dir)
    results = prewarm_story_cache(
        loader,
        quests=args.quests or DEFAULT_QUESTS,
        regions=args.regions or DEFAULT_REGIONS,
        phases=args.phases or DEFAULT_PHASES,
        refresh=args.refresh,
    )
    return 0 if results and all(result.success for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
