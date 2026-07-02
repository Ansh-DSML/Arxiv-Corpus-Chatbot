"""
arXiv Paper Collector — Core ML/DL/NN/AI Algorithm Papers
============================================================
Collects core algorithmic papers (not applied-domain papers) from arXiv
across four categories: dl, ml, neural_networks, ai (RAG techniques).

Resumable: every arXiv ID ever processed is tracked in a JSON manifest,
saved to disk after EVERY paper. If the script is killed mid-run — rate
limit, network drop, Ctrl+C, laptop sleep — re-running it picks up exactly
where it left off. No paper is ever re-downloaded or double-counted,
whether the interruption happened after 5 papers or 495.

Diversity via round-robin per-keyword quotas:
  - Each category has dozens of keywords. Left unconstrained, the first
    keyword or two in the list can absorb the entire --target-per-category
    quota before later keywords are ever searched (e.g. "transformer
    architecture" alone returning 100+ hits before "flash attention" gets
    a single query).
  - Instead, collection proceeds in rounds. Round 1 gives every keyword a
    fair-share cap of roughly target_count / num_keywords new downloads.
    Any keyword that can't fill its cap (genuinely has fewer matching
    papers in its top-100 relevance-ranked pool) is marked exhausted for
    this run. Round 2 recomputes the cap over only the still-active
    keywords, spreading the leftover quota across them. This repeats
    until the category target is hit or every keyword is exhausted.
  - Net effect: a keyword-diverse corpus first, with quota only spilling
    over to "hungrier" keywords once the others have genuinely run dry.

Rate-limit safe:
  - Search calls go through arxiv.Client(delay_seconds=3, num_retries=5),
    which is the library's built-in mechanism matching arXiv's documented
    courtesy limit of ~1 request per 3 seconds.
  - PDF downloads (a separate endpoint from search) get their own explicit
    throttle + exponential backoff, since a burst of consecutive downloads
    is the most common way to get soft-blocked.

Usage:
    python collect_arxiv_papers.py --target-per-category 150
    python collect_arxiv_papers.py --categories dl,ml --target-per-category 100
    python collect_arxiv_papers.py --resume-only   # only retry failed downloads
"""

import argparse
import json
import logging
import math
import re
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import arxiv
import requests

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arxiv_collector")


# ─────────────────────────────────────────────────────────
# Search config — one query per keyword, kept deliberately
# focused on CORE algorithms/methods, not applied use-cases.
# ─────────────────────────────────────────────────────────
CATEGORY_QUERIES = {
    "dl": {
        "arxiv_categories": ["cs.LG", "cs.NE"],
        "keywords": [
    "transformer architecture",
    "attention mechanism",
    "multi-head attention",
    "self-attention",
    "cross attention",
    "vision transformer",
    "masked autoencoder",
    "diffusion model",
    "latent diffusion",
    "score based generative model",
    "denoising diffusion probabilistic model",
    "generative adversarial network",
    "wasserstein gan",
    "conditional gan",
    "variational autoencoder",
    "autoencoder representation learning",
    "contrastive learning",
    "self supervised learning",
    "representation learning",
    "metric learning",
    "knowledge distillation",
    "mixture of experts",
    "sparse mixture of experts",
    "parameter efficient fine tuning",
    "low rank adaptation",
    "adapter tuning",
    "prompt tuning",
    "instruction tuning",
    "reinforcement learning from human feedback",
    "direct preference optimization",
    "chain of thought reasoning",
    "test time scaling",
    "normalization layer",
    "batch normalization",
    "layer normalization",
    "group normalization",
    "dropout regularization",
    "weight decay",
    "gradient descent optimization",
    "adam optimizer",
    "adamw optimizer",
    "learning rate scheduler",
    "curriculum learning",
    "deep residual learning",
    "feature pyramid network",
    "dense connectivity",
    "deep supervision",
    "token pruning",
    "efficient transformer",
    "flash attention",
    "long context transformer",
        ],
    },
    "ml": {
        "arxiv_categories": ["stat.ML", "cs.LG"],
        "keywords": [
            "support vector machine",
    "kernel method",
    "random forest",
    "decision tree algorithm",
    "gradient boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "ensemble learning",
    "bagging",
    "boosting algorithm",
    "stacking ensemble",
    "bayesian inference",
    "bayesian optimization",
    "gaussian process",
    "gaussian mixture model",
    "expectation maximization",
    "hidden markov model",
    "k nearest neighbor",
    "nearest neighbor search",
    "clustering algorithm",
    "hierarchical clustering",
    "spectral clustering",
    "dbscan clustering",
    "mean shift clustering",
    "dimensionality reduction",
    "principal component analysis",
    "independent component analysis",
    "linear discriminant analysis",
    "manifold learning",
    "t-sne",
    "umap",
    "feature selection",
    "feature engineering",
    "active learning",
    "semi supervised learning",
    "online learning",
    "incremental learning",
    "meta learning",
    "few shot learning",
    "zero shot learning",
    "multi task learning",
    "transfer learning",
    "probabilistic graphical model",
    "causal inference",
    "causal discovery",
    "outlier detection",
    "anomaly detection",
    "imbalanced learning",
        ],
    },
    "neural_networks": {
        "arxiv_categories": ["cs.NE"],
        "keywords": [
            "artificial neural network",
    "feedforward neural network",
    "multilayer perceptron",
    "backpropagation algorithm",
    "residual network",
    "highway network",
    "dense convolutional network",
    "convolutional neural network",
    "alexnet",
    "vgg network",
    "googlenet inception",
    "resnet",
    "densenet",
    "efficientnet",
    "mobilenet",
    "shufflenet",
    "convnext",
    "vision transformer",
    "swin transformer",
    "deit transformer",
    "masked autoencoder vision",
    "segment anything model",
    "object detection transformer",
    "yolo architecture",
    "faster rcnn",
    "mask rcnn",
    "retinanet",
    "unet segmentation",
    "deeplab segmentation",
    "recurrent neural network",
    "long short term memory",
    "gated recurrent unit",
    "bidirectional recurrent neural network",
    "sequence to sequence",
    "encoder decoder architecture",
    "transformer neural network",
    "bert architecture",
    "roberta architecture",
    "gpt architecture",
    "t5 transformer",
    "llama architecture",
    "mistral transformer",
    "graph neural network",
    "graph attention network",
    "graph convolutional network",
    "graph transformer",
    "spiking neural network",
    "capsule network",
    "echo state network",
    "reservoir computing",
    "neural architecture search",
    "activation function",
    "weight initialization",
    "skip connection",
    "attention neural network",
        ],
    },
    "ai": {
        "arxiv_categories": ["cs.CL", "cs.IR", "cs.LG"],
        "keywords": [
            "retrieval augmented generation",
    "retrieval augmented language model",
    "dense passage retrieval",
    "sparse retrieval",
    "hybrid retrieval",
    "vector search",
    "approximate nearest neighbor search",
    "semantic search",
    "embedding model",
    "embedding retrieval",
    "cross encoder reranking",
    "bi encoder retrieval",
    "reranking retrieval",
    "late interaction retrieval",
    "colbert retrieval",
    "knowledge graph retrieval",
    "graph rag",
    "adaptive retrieval",
    "multi hop retrieval",
    "query expansion",
    "query rewriting",
    "document chunking",
    "context compression",
    "retrieval optimization",
    "memory augmented transformer",
    "memory augmented language model",
    "agentic retrieval",
    "tool augmented language model",
    "retrieval planning",
    "retrieval benchmark",
    "long context language model",
    "context window optimization",
    "vector database indexing",
    "embedding compression",
    "rag evaluation",
    "hallucination mitigation",
    "grounded generation",
    "fact verification language model",
    "knowledge editing language model",
    "retrieval calibration",
    "language model reasoning",
    "chain of thought",
    "tree of thoughts",
    "reasoning language model",
        ],
    },
}

# Papers whose title/abstract clearly signal an *applied* domain rather
# than a core algorithm/method contribution — filtered out even if they
# matched a keyword above.
APPLIED_EXCLUDE_KEYWORDS = [
    "medical imaging", "clinical", "diagnosis", "disease", "drug discovery",
    "protein structure", "genome", "gene expression", "autonomous driving",
    "self-driving", "remote sensing", "satellite imagery", "agricultur",
    "financial market", "stock price", "fraud detection", "traffic prediction",
    "speech recognition system for", "recommendation system for",
]

RESULTS_PER_QUERY = 100   # candidate pool per keyword before filtering
MAX_TITLE_LEN_LOG = 70


@dataclass
class PaperRecord:
    arxiv_id: str
    title: str
    category: str
    status: str              # "downloaded" | "failed" | "skipped_applied"
    pdf_path: Optional[str] = None
    matched_keyword: Optional[str] = None
    attempts: int = 0


class Manifest:
    """Tracks every arXiv ID ever seen, across ALL categories, so the
    collector is fully resumable and never double-downloads a paper."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, PaperRecord] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return

        text = self.path.read_text()
        if not text.strip():
            # Empty file (0 bytes or whitespace-only) — nothing to lose,
            # just start fresh rather than crashing the whole run.
            log.warning(f"Manifest at {self.path} is empty — starting a fresh manifest. "
                        f"Already-downloaded PDFs on disk will be re-detected automatically "
                        f"the next time their search result comes up.")
            return

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            # Manifest exists but isn't valid JSON (truncated write, manual edit
            # gone wrong, etc.). Don't take down the whole run over this — back
            # up the bad file so nothing is silently lost, and start fresh.
            backup = self.path.with_suffix(f".corrupt-{int(time.time())}.json")
            self.path.replace(backup)
            log.error(f"Manifest at {self.path} was not valid JSON ({e}). "
                      f"Backed up the unreadable file to {backup} and starting a fresh "
                      f"manifest. Already-downloaded PDFs on disk will be re-detected "
                      f"automatically the next time their search result comes up.")
            return

        self.records = {k: PaperRecord(**v) for k, v in raw.items()}
        log.info(f"Loaded manifest with {len(self.records)} existing records")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in self.records.items()}, indent=2))
        tmp.replace(self.path)  # atomic write — a crash mid-save can never
                                  # corrupt the manifest, worst case you lose
                                  # the write in progress, not the whole file

    def seen(self, arxiv_id: str) -> bool:
        return arxiv_id in self.records and self.records[arxiv_id].status == "downloaded"

    def count_downloaded(self, category: str) -> int:
        return sum(
            1 for r in self.records.values()
            if r.category == category and r.status == "downloaded"
        )

    def count_downloaded_for_keyword(self, category: str, keyword: str) -> int:
        """Downloads attributed to one specific keyword within a category.
        Used to enforce the per-keyword fair-share cap during round-robin
        collection, so no single keyword can monopolize a category's quota."""
        return sum(
            1 for r in self.records.values()
            if r.category == category
            and r.matched_keyword == keyword
            and r.status == "downloaded"
        )

    def upsert(self, record: PaperRecord):
        self.records[record.arxiv_id] = record
        self.save()  # persisted after EVERY record — a crash never loses
                      # more than the single in-flight download


def is_applied_paper(title: str, abstract: str) -> bool:
    text = f"{title} {abstract}".lower()
    return any(kw in text for kw in APPLIED_EXCLUDE_KEYWORDS)


def clean_arxiv_id(entry_id: str) -> str:
    """arxiv.Result.entry_id looks like 'http://arxiv.org/abs/2301.12345v2'.
    Strip to the stable, version-less ID so re-runs match consistently even
    if a paper gets revised (new version) between runs."""
    raw = entry_id.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", raw)


def download_with_backoff(result: arxiv.Result, out_path: Path, max_retries: int = 5) -> bool:
    """Downloads a single PDF with exponential backoff. Never raises —
    returns False on exhausted retries so the caller can log & move on
    instead of the whole run dying on one bad paper.

    Downloads directly via result.pdf_url + requests rather than the
    library's Result.download_pdf() helper — that helper was removed in
    arxiv>=3.0.0 (the library's own docs now recommend this same approach),
    and downloading straight from pdf_url also means this keeps working
    even if the library's convenience API changes again later."""
    delay = 3.0
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(result.pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            log.warning(f"Download attempt {attempt}/{max_retries} failed "
                        f"({e.__class__.__name__}: {e}) — retrying in {delay:.0f}s")
            out_path.unlink(missing_ok=True)  # clear any partial/corrupt file
            time.sleep(delay)
            delay = min(delay * 2, 60)  # cap backoff at 60s
    log.error(f"Giving up on {out_path.name} after {max_retries} attempts")
    return False


def process_result(
    result: arxiv.Result,
    category: str,
    keyword: str,
    manifest: Manifest,
    category_dir: Path,
    api_delay: float,
):
    """Handles one search result: dedup check, applied-domain filter,
    download, manifest update. Returns nothing — all side effects go
    through the manifest."""
    arxiv_id = clean_arxiv_id(result.entry_id)

    # Global resumability + cross-category dedup: skip anything already
    # downloaded, in ANY category, so no paper is ever pulled twice or
    # double-counted toward two targets.
    if manifest.seen(arxiv_id):
        return

    title = result.title.strip().replace("\n", " ")
    abstract = result.summary.strip().replace("\n", " ")

    if is_applied_paper(title, abstract):
        manifest.upsert(PaperRecord(
            arxiv_id=arxiv_id, title=title, category=category,
            status="skipped_applied", matched_keyword=keyword,
        ))
        return

    safe_title = re.sub(r"[^\w\s-]", "", title)[:80].strip().replace(" ", "_")
    filename = f"{arxiv_id}_{safe_title}.pdf"
    pdf_path = category_dir / filename

    if pdf_path.exists():
        # File already on disk from a prior run but the manifest write
        # didn't complete — trust the file, don't re-download.
        manifest.upsert(PaperRecord(
            arxiv_id=arxiv_id, title=title, category=category,
            status="downloaded", pdf_path=str(pdf_path),
            matched_keyword=keyword,
        ))
        return

    log.info(f"[{category}] downloading {arxiv_id}: {title[:MAX_TITLE_LEN_LOG]}...")

    success = download_with_backoff(result, pdf_path)

    manifest.upsert(PaperRecord(
        arxiv_id=arxiv_id, title=title, category=category,
        status="downloaded" if success else "failed",
        pdf_path=str(pdf_path) if success else None,
        matched_keyword=keyword,
        attempts=1,
    ))

    # Throttle between PDF downloads — a different endpoint from search,
    # still courteous to stay well clear of any limits.
    time.sleep(api_delay)


def run_keyword_search(
    category: str,
    keyword: str,
    search: arxiv.Search,
    client: arxiv.Client,
    manifest: Manifest,
    category_dir: Path,
    category_target: int,
    keyword_target: int,
    api_delay: float,
    max_query_retries: int = 4,
):
    """Iterates search results for one keyword, with real retry/backoff
    around the iteration itself — not just around creating the generator.

    Stops on whichever comes first:
      - the category's overall quota is met (category_target), or
      - this keyword's fair-share cap for the current round is met
        (keyword_target) — this is what keeps one keyword from draining
        the whole category before others get a turn.

    The arxiv library's HTTP requests happen lazily, INSIDE the for-loop
    as you pull each page, so a try/except around client.results(search)
    alone (which just builds a generator) never actually catches a 429/503
    — only wrapping the iteration does. On failure we back off and restart
    the same keyword query from page 0; already-downloaded papers are
    skipped near-instantly via the manifest, so this costs a little
    redundant iteration, never a redundant download.
    """
    attempt = 0
    while attempt < max_query_retries:
        attempt += 1
        try:
            for result in client.results(search):
                if manifest.count_downloaded(category) >= category_target:
                    return
                if manifest.count_downloaded_for_keyword(category, keyword) >= keyword_target:
                    return
                process_result(result, category, keyword, manifest, category_dir, api_delay)
            return  # exhausted this keyword's results normally
        except arxiv.HTTPError as e:
            wait = 45.0 if "429" in str(e) else 20.0
            log.warning(f"[{category}] arXiv API error on \"{keyword}\" "
                        f"(attempt {attempt}/{max_query_retries}): {e} "
                        f"— backing off {wait:.0f}s")
            time.sleep(wait)
        except Exception as e:
            log.warning(f"[{category}] unexpected error on \"{keyword}\" "
                        f"(attempt {attempt}/{max_query_retries}): "
                        f"{e.__class__.__name__}: {e} — backing off 20s")
            time.sleep(20.0)

    log.error(f"[{category}] giving up on \"{keyword}\" after {max_query_retries} "
              f"attempts — moving to next keyword (progress so far is saved)")


def collect_category(
    category: str,
    target_count: int,
    output_dir: Path,
    manifest: Manifest,
    client: arxiv.Client,
    api_delay: float,
):
    cfg = CATEGORY_QUERIES[category]
    keywords = cfg["keywords"]
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    already = manifest.count_downloaded(category)
    log.info(f"[{category}] starting with {already}/{target_count} already downloaded")

    if already >= target_count:
        log.info(f"[{category}] target already met, skipping")
        return

    cat_filter = " OR ".join(f"cat:{c}" for c in cfg["arxiv_categories"])

    # Round-robin collection: give every keyword a fair-share cap of the
    # remaining quota each round. A keyword that can't fill its cap (its
    # top-RESULTS_PER_QUERY relevance pool has nothing new left to give)
    # is marked exhausted and dropped from later rounds; its leftover
    # share gets redistributed across whatever keywords are still active.
    exhausted_keywords: set[str] = set()
    round_num = 0

    while (
        manifest.count_downloaded(category) < target_count
        and len(exhausted_keywords) < len(keywords)
    ):
        round_num += 1
        active_keywords = [k for k in keywords if k not in exhausted_keywords]
        remaining_needed = target_count - manifest.count_downloaded(category)
        per_keyword_cap = max(1, math.ceil(remaining_needed / len(active_keywords)))

        log.info(f"[{category}] round {round_num}: {len(active_keywords)} active "
                 f"keyword(s), {remaining_needed} paper(s) still needed, "
                 f"cap {per_keyword_cap}/keyword this round")

        for keyword in active_keywords:
            if manifest.count_downloaded(category) >= target_count:
                break

            before = manifest.count_downloaded_for_keyword(category, keyword)
            keyword_target = before + per_keyword_cap

            query = f"({cat_filter}) AND abs:({keyword})"
            search = arxiv.Search(
                query=query,
                max_results=RESULTS_PER_QUERY,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            log.info(f"[{category}] searching: \"{keyword}\" "
                     f"(has {before}, capped at {keyword_target} this round)")

            run_keyword_search(
                category, keyword, search, client, manifest,
                category_dir, target_count, keyword_target, api_delay,
            )

            after = manifest.count_downloaded_for_keyword(category, keyword)
            if after == before:
                # No new downloads for this keyword even with room under
                # its cap to take them — its candidate pool is genuinely
                # dry (or every remaining candidate is applied-domain /
                # a permanently failed download). Don't keep re-querying
                # it every round.
                exhausted_keywords.add(keyword)
                log.info(f"[{category}] \"{keyword}\" exhausted "
                         f"({after} total from this keyword)")

            # Pause between DIFFERENT keyword searches too — the client's
            # own delay_seconds only throttles pages *within* one search,
            # not the gap between separate Search() calls.
            time.sleep(api_delay)

    final = manifest.count_downloaded(category)
    log.info(f"[{category}] finished: {final}/{target_count} downloaded "
             f"across {len(keywords) - len(exhausted_keywords)}/{len(keywords)} "
             f"keywords with remaining supply "
             f"({round_num} round(s))")


def retry_failed(manifest: Manifest, output_dir: Path):
    """Re-attempt every record marked 'failed' from a previous run."""
    failed = [r for r in manifest.records.values() if r.status == "failed"]
    if not failed:
        log.info("No failed downloads to retry")
        return

    log.info(f"Retrying {len(failed)} previously failed downloads")
    search = arxiv.Search(id_list=[r.arxiv_id for r in failed])
    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=5)

    for result in client.results(search):
        arxiv_id = clean_arxiv_id(result.entry_id)
        record = manifest.records.get(arxiv_id)
        if not record:
            continue

        category_dir = output_dir / record.category
        category_dir.mkdir(parents=True, exist_ok=True)
        safe_title = re.sub(r"[^\w\s-]", "", record.title)[:80].strip().replace(" ", "_")
        pdf_path = category_dir / f"{arxiv_id}_{safe_title}.pdf"

        success = download_with_backoff(result, pdf_path)
        record.status = "downloaded" if success else "failed"
        record.pdf_path = str(pdf_path) if success else None
        record.attempts += 1
        manifest.upsert(record)
        time.sleep(3.0)


def main():
    parser = argparse.ArgumentParser(description="Collect core ML/DL/NN/AI papers from arXiv")
    parser.add_argument("--categories", type=str, default="dl,ml,neural_networks,ai",
                         help="comma-separated subset of: dl,ml,neural_networks,ai")
    parser.add_argument("--target-per-category", type=int, default=500,
                         help="number of papers to collect per category")
    parser.add_argument("--output-dir", type=str, default="../data/raw",
                         help="root output directory (category subfolders created inside)")
    parser.add_argument("--manifest", type=str, default="../data/manifest.json",
                         help="path to the resumability manifest JSON file")
    parser.add_argument("--api-delay", type=float, default=3.0,
                         help="seconds to sleep between PDF downloads (courtesy rate limit)")
    parser.add_argument("--resume-only", action="store_true",
                         help="only retry previously failed downloads, don't search for new ones")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest = Manifest(Path(args.manifest))

    # arxiv.Client handles the rate-limit/retry contract for *search*
    # requests (delay_seconds=3 matches arXiv's documented courtesy limit
    # of one request per 3 seconds; num_retries handles transient errors
    # like UnexpectedEmptyPageError automatically).
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)

    def handle_interrupt(signum, frame):
        log.warning("Interrupted — manifest is already saved through the last "
                     "completed paper. Just re-run this script to resume.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    if args.resume_only:
        retry_failed(manifest, output_dir)
        return

    categories = [c.strip() for c in args.categories.split(",")]
    for category in categories:
        if category not in CATEGORY_QUERIES:
            log.error(f"Unknown category '{category}', skipping. "
                      f"Valid: {list(CATEGORY_QUERIES.keys())}")
            continue
        collect_category(
            category=category,
            target_count=args.target_per_category,
            output_dir=output_dir,
            manifest=manifest,
            client=client,
            api_delay=args.api_delay,
        )

    log.info("─" * 50)
    total = 0
    for category in CATEGORY_QUERIES:
        n = manifest.count_downloaded(category)
        total += n
        log.info(f"  {category:18s}: {n} papers")
    log.info(f"  {'TOTAL':18s}: {total} papers")

    failed_count = sum(1 for r in manifest.records.values() if r.status == "failed")
    if failed_count:
        log.info(f"\n{failed_count} downloads failed — re-run with --resume-only to retry them")


if __name__ == "__main__":
    main()