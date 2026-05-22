from python.path_utils import file_paths_match


NOT_FOUND_PREFIX = "в предоставленных документах нет информации"


def retrieval_hit(hit_files: list[str], source_file: str) -> bool:
    for path in hit_files:
        if file_paths_match(path, source_file):
            return True
    return False


def answer_hit_loose(answer: str, must_contain: list[str]) -> bool:
    low = answer.lower()
    hits = sum(1 for token in must_contain if token.lower() in low)
    return hits >= max(1, len(must_contain) // 2)


def answer_hit_strict(answer: str, must_contain: list[str]) -> bool:
    low = answer.lower()
    return all(token.lower() in low for token in must_contain)


def answer_grounded(answer: str) -> bool:
    return not answer.lower().strip().startswith(NOT_FOUND_PREFIX)
