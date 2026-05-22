import re


_FILENAME_PREFIX = re.compile(r"^\[[^\]]+\]\s*")


def canonical_file_key(path: str) -> str:
    s = path.replace("\\", "/").strip().lower()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    s = _FILENAME_PREFIX.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s


def file_paths_match(hit_path: str, source_file: str) -> bool:
    left = canonical_file_key(hit_path)
    right = canonical_file_key(source_file)
    return left == right
