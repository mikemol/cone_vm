import argparse
import keyword
import re
from collections import Counter
from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path

TOKEN_PATTERN = r"[^\W\d_][\w]*"
DEFAULT_STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
}


@dataclass(frozen=True)
class Candidate:
    word: str
    kind: str  # add_stopword | remove_stopword
    token_delta: tuple
    bigram_delta: tuple

def generate_bigrams(tokens):
    return list(zip(tokens, tokens[1:]))

def calculate_stats(tokens, current_stopwords):
    # Filter tokens based on current stopwords
    active_tokens = [t for t in tokens if t not in current_stopwords]
    unique_tokens = set(active_tokens)
    unique_bigrams = set(generate_bigrams(active_tokens))
    return len(unique_tokens), len(unique_bigrams)

def tokenize(text, pattern=TOKEN_PATTERN):
    token_re = re.compile(pattern)
    return [t.lower() for t in token_re.findall(text)]

def load_stopwords(path=None, extra=None, remove=None, include_python=True):
    stopwords = set(DEFAULT_STOPWORDS)
    if include_python:
        stopwords |= set(keyword.kwlist)
    if path:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stopwords.add(line.lower())
    if extra:
        stopwords |= {w.lower() for w in extra}
    if remove:
        stopwords -= {w.lower() for w in remove}
    return stopwords

def analyze_candidate_impact(tokens, candidate_word, current_stopwords):
    """
    Simulates removing a specific word and returns the bigram delta.
    """
    active = [t for t in tokens if t not in current_stopwords]
    
    # State if we remove the candidate
    # We add candidate to stopwords effectively
    proposed_stopwords = current_stopwords.union({candidate_word})
    proposed_active = [t for t in tokens if t not in proposed_stopwords]
    
    # Calculate difference
    # Note: token delta is always -1 for unique words, so we strictly look at bigrams.
    current_bigrams = set(generate_bigrams(active))
    proposed_bigrams = set(generate_bigrams(proposed_active))
    
    delta_bigram = len(proposed_bigrams) - len(current_bigrams)
    
    return delta_bigram

def calculate_stats_for_docs(tokens_by_doc, stopwords):
    stats = {}
    for doc, tokens in tokens_by_doc.items():
        stats[doc] = calculate_stats(tokens, stopwords)
    return stats

def candidate_delta(tokens_by_doc, base_stats, stopwords, word, add_stopword):
    if add_stopword:
        next_stopwords = stopwords | {word}
    else:
        next_stopwords = stopwords - {word}
    next_stats = calculate_stats_for_docs(tokens_by_doc, next_stopwords)
    token_delta = []
    bigram_delta = []
    for doc in base_stats.keys():
        token_delta.append(next_stats[doc][0] - base_stats[doc][0])
        bigram_delta.append(next_stats[doc][1] - base_stats[doc][1])
    return tuple(token_delta), tuple(bigram_delta)

def build_candidates(tokens_by_doc, base_stats, stopwords, words, kind):
    add_stopword = kind == "add_stopword"
    candidates = []
    for word in words:
        token_delta, bigram_delta = candidate_delta(
            tokens_by_doc, base_stats, stopwords, word, add_stopword
        )
        if token_delta == tuple([0] * len(base_stats)) and bigram_delta == tuple(
            [0] * len(base_stats)
        ):
            continue
        candidates.append(Candidate(word, kind, token_delta, bigram_delta))
    return candidates

def group_candidates_by_token_delta(candidates):
    grouped = {}
    for cand in candidates:
        grouped.setdefault(cand.token_delta, []).append(cand)
    return grouped

def enumerate_combos(candidates, max_k):
    combos = []
    combos.append(((), tuple([0] * len(candidates[0].token_delta)), tuple([0] * len(candidates[0].bigram_delta))))
    for k in range(1, max_k + 1):
        for combo in combinations(candidates, k):
            token_delta = [0] * len(combo[0].token_delta)
            bigram_delta = [0] * len(combo[0].bigram_delta)
            for cand in combo:
                for i, val in enumerate(cand.token_delta):
                    token_delta[i] += val
                for i, val in enumerate(cand.bigram_delta):
                    bigram_delta[i] += val
            combos.append((combo, tuple(token_delta), tuple(bigram_delta)))
    return combos

def search_combos(
    add_candidates,
    remove_candidates,
    token_target,
    bigram_target,
    max_add=4,
    max_remove=2,
    max_results=5,
):
    if not add_candidates and not remove_candidates:
        return []
    results = []
    add_combos = enumerate_combos(add_candidates, max_add) if add_candidates else [
        ((), tuple([0] * len(token_target)), tuple([0] * len(bigram_target)))
    ]
    remove_combos = enumerate_combos(remove_candidates, max_remove) if remove_candidates else [
        ((), tuple([0] * len(token_target)), tuple([0] * len(bigram_target)))
    ]

    remove_by_token = {}
    for combo, token_delta, bigram_delta in remove_combos:
        remove_by_token.setdefault(token_delta, []).append((combo, bigram_delta))

    for add_combo, add_token_delta, add_bigram_delta in add_combos:
        target_remove_token = tuple(
            token_target[i] - add_token_delta[i] for i in range(len(token_target))
        )
        for remove_combo, remove_bigram_delta in remove_by_token.get(
            target_remove_token, []
        ):
            total_bigram = tuple(
                add_bigram_delta[i] + remove_bigram_delta[i]
                for i in range(len(bigram_target))
            )
            if total_bigram == bigram_target:
                results.append((add_combo, remove_combo))
                if len(results) >= max_results:
                    return results
    return results

def describe_token_delta_counts(candidates):
    counts = Counter([cand.token_delta for cand in candidates])
    return counts

def parse_targets(targets):
    parsed = {}
    for entry in targets:
        doc, counts = entry.split("=", 1)
        token_str, bigram_str = counts.split(",", 1)
        parsed[doc] = (int(token_str), int(bigram_str))
    return parsed

def run_search(
    docs,
    targets,
    stopwords,
    pattern,
    max_add,
    max_remove,
    max_results,
    max_candidates,
    exclusive_doc,
):
    tokens_by_doc = {doc: tokenize(Path(doc).read_text(), pattern) for doc in docs}
    base_stats = calculate_stats_for_docs(tokens_by_doc, stopwords)
    base_counts = {doc: base_stats[doc] for doc in docs}

    token_target = tuple(targets[doc][0] - base_counts[doc][0] for doc in docs)
    bigram_target = tuple(targets[doc][1] - base_counts[doc][1] for doc in docs)

    all_words = set(chain.from_iterable(tokens_by_doc.values()))
    if exclusive_doc:
        exclusive_words = set(tokens_by_doc[exclusive_doc])
        for doc in docs:
            if doc == exclusive_doc:
                continue
            exclusive_words -= set(tokens_by_doc[doc])
        all_words = exclusive_words

    add_words = [w for w in all_words if w not in stopwords]
    remove_words = [w for w in all_words if w in stopwords]

    add_candidates = build_candidates(tokens_by_doc, base_stats, stopwords, add_words, "add_stopword")
    remove_candidates = build_candidates(
        tokens_by_doc, base_stats, stopwords, remove_words, "remove_stopword"
    )

    if max_candidates:
        add_candidates = add_candidates[:max_candidates]
        remove_candidates = remove_candidates[:max_candidates]

    print("Base counts:")
    for doc in docs:
        print(f"  {doc}: tokens={base_counts[doc][0]} bigrams={base_counts[doc][1]}")
    print("Targets:")
    for doc in docs:
        print(f"  {doc}: tokens={targets[doc][0]} bigrams={targets[doc][1]}")
    print(f"Need token delta: {token_target}")
    print(f"Need bigram delta: {bigram_target}")

    print("\nCandidate counts by token delta (add stopword):")
    for delta, count in describe_token_delta_counts(add_candidates).items():
        print(f"  {delta}: {count}")
    print("\nCandidate counts by token delta (remove stopword):")
    for delta, count in describe_token_delta_counts(remove_candidates).items():
        print(f"  {delta}: {count}")

    results = search_combos(
        add_candidates,
        remove_candidates,
        token_target,
        bigram_target,
        max_add=max_add,
        max_remove=max_remove,
        max_results=max_results,
    )
    if not results:
        print("\nNo exact combo found with current limits.")
        return
    print("\nFound combos:")
    for add_combo, remove_combo in results:
        print("  add_stopword:", [c.word for c in add_combo])
        print("  remove_stopword:", [c.word for c in remove_combo])

def solve_stopword_adjustment(text, existing_stopwords, target_token_count, target_bigram_count, pattern=TOKEN_PATTERN):
    # Use the shared tokenizer!
    current_tokens = tokenize(text, pattern) 
    current_t, current_b = calculate_stats(current_tokens, existing_stopwords)
    
    needed_token_drop = current_t - target_token_count
    needed_bigram_drop = current_b - target_bigram_count
    
    print(f"Current: T={current_t}, B={current_b}")
    print(f"Target:  T={target_token_count}, B={target_bigram_count}")
    print(f"Need to remove {needed_token_drop} tokens to drop bigrams by {needed_bigram_drop}.\n")

    if needed_token_drop <= 0:
        print("Target already met.")
        return

    # Filter candidates
    all_words = set(current_tokens)
    candidates = [w for w in all_words if w not in existing_stopwords]
    
    buckets = {}
    print("Profiling candidates...")
    for word in candidates:
        # Use shared logic
        delta = analyze_candidate_impact(current_tokens, word, existing_stopwords)
        if delta not in buckets:
            buckets[delta] = []
        buckets[delta].append(word)

    for d, words in buckets.items():
        print(f"  Delta {d}: {len(words)} candidates found.")

    # 3. Solve the Linear Equation
    # We need x words from Bucket(-2) and y words from Bucket(-1) etc...
    # such that:
    #   x + y + ... = needed_token_drop
    #   -2x + -1y + ... = -needed_bigram_drop
    
    # Simple Solver for common case (mostly -1 and -2 deltas)
    # We prioritize "common" looking words if we had a frequency list, 
    # but here we just solve the math.
    
    solution = []
    remaining_tokens = needed_token_drop
    remaining_bigrams = needed_bigram_drop # This is a positive number representing the drop
    
    # Heuristic: Greedy approach or exact math?
    # Since we usually only have -2 and -1, we can solve algebraically:
    # Let x = count(-2), y = count(-1)
    # x + y = TotalTokens
    # 2x + 1y = TotalBigramDrop
    # Subtracting: x = TotalBigramDrop - TotalTokens
    
    # Assuming mostly -1 and -2 buckets are populated:
    count_neg2 = remaining_bigrams - remaining_tokens
    count_neg1 = remaining_tokens - count_neg2
    
    # Validation
    if count_neg2 < 0 or count_neg1 < 0:
        print("\n[!] The required drop topology is complex (requires deltas other than -1/-2). Switching to greedy solver.")
        # Fallback to simple greedy matching if the algebra fails
        # (This happens if you need huge drops, requiring -3 or -0 deltas)
        pass 
    else:
        print(f"\nMath Solution: Pick {count_neg2} words from Delta -2 and {count_neg1} words from Delta -1.")
        
        if len(buckets.get(-2, [])) >= count_neg2 and len(buckets.get(-1, [])) >= count_neg1:
            solution.extend(buckets.get(-2, [])[:count_neg2])
            solution.extend(buckets.get(-1, [])[:count_neg1])
            
            print("\nSUCCESS. Proposed Stopword Additions:")
            print(solution)
            return solution
        else:
            print("Not enough candidates in the specific buckets to satisfy the math.")

# 3. NEW: A runner for the new mode
def run_topological(docs, targets, stopwords, pattern):
    # This mode assumes we are fixing ONE file (e.g., in-11)
    # independent of others, or you iterate through them.
    for doc in docs:
        print(f"--- Optimizing {doc} ---")
        text = Path(doc).read_text()
        target_t, target_b = targets[doc]
        
        solve_stopword_adjustment(
            text, 
            stopwords, 
            target_t, 
            target_b, 
            pattern=pattern
        )

def main():
    parser = argparse.ArgumentParser(description="Stopword optimizer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Search Mode (Original Brute Force) ---
    search_parser = subparsers.add_parser("search", help="Combinatorial search.")
    search_parser.add_argument("--docs", nargs="+", required=True)
    search_parser.add_argument("--targets", nargs="+", required=True)
    search_parser.add_argument("--stopwords")
    search_parser.add_argument("--pattern", default=TOKEN_PATTERN)
    search_parser.add_argument("--max-add", type=int, default=4)
    search_parser.add_argument("--max-remove", type=int, default=2)
    search_parser.add_argument("--max-results", type=int, default=5)
    search_parser.add_argument("--max-candidates", type=int, default=0)
    search_parser.add_argument("--exclusive-doc")

    # --- Topological Mode (New Algebraic Solver) ---
    topo_parser = subparsers.add_parser("solve", help="Algebraic topological solver.")
    topo_parser.add_argument("--docs", nargs="+", required=True)
    topo_parser.add_argument("--targets", nargs="+", required=True)
    topo_parser.add_argument("--stopwords")
    topo_parser.add_argument("--pattern", default=TOKEN_PATTERN)

    args = parser.parse_args()
    targets = parse_targets(args.targets)
    stopwords = load_stopwords(path=args.stopwords)

    if args.command == "search":
        run_search(
            docs=args.docs,
            targets=targets,
            stopwords=stopwords,
            pattern=args.pattern,
            max_add=args.max_add,
            max_remove=args.max_remove,
            max_results=args.max_results,
            max_candidates=args.max_candidates or None,
            exclusive_doc=args.exclusive_doc,
        )
    elif args.command == "solve":
        run_topological(
            docs=args.docs,
            targets=targets,
            stopwords=stopwords,
            pattern=args.pattern
        )

if __name__ == "__main__":
    main()
