"""
Crawl files from a GitLab repository using the GitLab API.

Supports gitlab.com and self-hosted GitLab instances.
Uses the same interface as crawl_github_files for consistency.
"""

import os
import time
import fnmatch
import requests
from typing import Union, Set, List, Dict, Any
from urllib.parse import urlparse, quote


def crawl_gitlab_files(
    repo_url: str,
    token: str = None,
    max_file_size: int = 1 * 1024 * 1024,  # 1 MB
    use_relative_paths: bool = False,
    include_patterns: Union[str, Set[str]] = None,
    exclude_patterns: Union[str, Set[str]] = None,
):
    """
    Crawl files from a GitLab repository.

    Args:
        repo_url: URL of the GitLab repository (e.g. https://gitlab.com/group/project or
                  https://gitlab.com/group/project/-/tree/main)
        token: GitLab personal access token. Required for private repos.
               Can be set via GITLAB_TOKEN environment variable.
        max_file_size: Maximum file size in bytes to download (default: 1 MB).
        use_relative_paths: If True, paths are relative to the specified subdirectory.
        include_patterns: Glob patterns for files to include (e.g. {"*.py", "*.md"}).
        exclude_patterns: Glob patterns for files to exclude.

    Returns:
        dict: {"files": {path: content}, "stats": {...}}
    """
    if include_patterns and isinstance(include_patterns, str):
        include_patterns = {include_patterns}
    if exclude_patterns and isinstance(exclude_patterns, str):
        exclude_patterns = {exclude_patterns}

    def should_include_file(file_path: str, file_name: str) -> bool:
        if not include_patterns:
            include_file = True
        else:
            include_file = any(
                fnmatch.fnmatch(file_name, p) for p in include_patterns
            )
        if exclude_patterns and include_file:
            exclude_file = any(
                fnmatch.fnmatch(file_path, p) for p in exclude_patterns
            )
            return not exclude_file
        return include_file

    parsed = urlparse(repo_url)
    host = parsed.netloc or "gitlab.com"
    path = parsed.path.strip("/")
    path_parts = path.split("/")

    if not path_parts:
        raise ValueError(f"Invalid GitLab URL: {repo_url}")

    # Project path is everything before "-" (GitLab uses "-" for /-/tree/...)
    try:
        dash_idx = path_parts.index("-")
        project_path = "/".join(path_parts[:dash_idx])
        rest = path_parts[dash_idx + 1 :]
    except ValueError:
        project_path = "/".join(path_parts)
        rest = []

    # Parse ref and subpath from rest: ["tree", "branch", "sub", "path"] or []
    ref = "HEAD"
    specific_path = ""
    if len(rest) >= 2 and rest[0] == "tree":
        ref = rest[1]
        if len(rest) > 2:
            specific_path = "/".join(rest[2:])

    project_id_encoded = quote(project_path, safe="")
    api_base = f"https://{host}/api/v4"
    if not repo_url.startswith("http"):
        api_base = f"https://{host}/api/v4"

    headers = {}
    if token:
        headers["PRIVATE-TOKEN"] = token

    # 1) List repository tree (recursive)
    tree_url = f"{api_base}/projects/{project_id_encoded}/repository/tree"
    params = {"recursive": "true", "ref": ref, "per_page": 100}
    all_items = []
    page = 1
    while True:
        params["page"] = page
        r = requests.get(tree_url, headers=headers, params=params, timeout=(30, 30))
        if r.status_code == 401:
            print(
                "GitLab 401: Invalid or missing token. Set GITLAB_TOKEN for private repos."
            )
            return {"files": {}, "stats": {"error": "Unauthorized"}}
        if r.status_code == 404:
            print(
                "GitLab 404: Project not found or no access. Check URL and token."
            )
            return {"files": {}, "stats": {"error": "Not found"}}
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 60))
            print(f"GitLab rate limit. Waiting {wait}s...")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            print(f"GitLab tree API error: {r.status_code} - {r.text[:500]}")
            return {"files": {}, "stats": {"error": r.text[:200]}}

        data = r.json()
        if not data:
            break
        all_items.extend(data)
        if len(data) < 100:
            break
        page += 1

    # 2) Collect blob paths (files only)
    blob_paths = []
    for item in all_items:
        if item.get("type") != "blob":
            continue
        path_item = item.get("path", "")
        if use_relative_paths and specific_path and path_item.startswith(specific_path):
            rel = path_item[len(specific_path) :].lstrip("/")
        else:
            rel = path_item
        if not should_include_file(rel, item.get("name", "")):
            continue
        blob_paths.append((path_item, rel))

    # 3) Fetch raw content for each file
    files = {}
    skipped_files = []
    for path_item, rel_path in blob_paths:
        file_path_encoded = quote(path_item, safe="")
        raw_url = (
            f"{api_base}/projects/{project_id_encoded}/repository/files/{file_path_encoded}/raw"
        )
        r = requests.get(
            raw_url, headers=headers, params={"ref": ref}, timeout=(30, 30)
        )
        if r.status_code != 200:
            print(f"Skip {rel_path}: HTTP {r.status_code}")
            skipped_files.append((rel_path, 0))
            continue
        content = r.text
        size = len(content.encode("utf-8"))
        if size > max_file_size:
            skipped_files.append((rel_path, size))
            print(f"Skipping {rel_path}: size {size} exceeds limit {max_file_size}")
            continue
        files[rel_path] = content
        print(f"Downloaded: {rel_path} ({size} bytes)")

    return {
        "files": files,
        "stats": {
            "downloaded_count": len(files),
            "skipped_count": len(skipped_files),
            "skipped_files": skipped_files,
            "base_path": specific_path if use_relative_paths else None,
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns,
        },
    }


if __name__ == "__main__":
    token = os.environ.get("GITLAB_TOKEN")
    url = "https://gitlab.com/group/project"
    result = crawl_gitlab_files(
        url,
        token=token,
        max_file_size=500 * 1024,
        include_patterns={"*.py", "*.md"},
    )
    print("Files:", len(result["files"]))
    print("Stats:", result["stats"])
