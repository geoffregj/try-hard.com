#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║              YTDL v1.1 — Terminal Edition                ║
║           Audio / Video Downloader (yt-dlp)              ║
╠══════════════════════════════════════════════════════════╣
║  Engine    : yt-dlp CLI (subprocess — never breaks on     ║
║              yt-dlp Python API changes)                   ║
║  Audio     : mp3, quality best/good/low → ~/Music          ║
║  Video     : mp4, quality best/1080p/720p/480p/360p        ║
║              → ~/Movies                                    ║
║  Search    : type a query instead of a link, pick from    ║
║              a numbered list of results                   ║
╚══════════════════════════════════════════════════════════╝
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

os.system("clear")

# ══════════════════════════════════════════════════════════
# PATHS & COLOURS
# ══════════════════════════════════════════════════════════
HOME = os.path.expanduser("~")
MUSIC_DIR = os.path.join(HOME, "Music")
MOVIES_DIR = os.path.join(HOME, "Movies")
CONFIG_FILE = os.path.join(HOME, "ytdlconfig.json")

R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
CYN = "\033[96m"; YLW = "\033[93m"; GRN = "\033[92m"
RED = "\033[91m"; MAG = "\033[95m"; WHT = "\033[97m"
BLU = "\033[94m"

AUDIO_QUALITIES = {"best": "0", "good": "5", "low": "9"}
VIDEO_FORMATS = {
    "best": "bestvideo+bestaudio/best",
    "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
    "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
}

SEARCH_RESULTS = 8


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def section(title):
    print(f"\n  {BLU}{'═'*50}{R}\n  {B}{WHT}  {title}{R}\n  {BLU}{'═'*50}{R}\n")


def log(msg, colour=DIM):
    print(f"  {colour}[{datetime.now().strftime('%H:%M:%S')}] {msg}{R}")


def ask(prompt, default=None):
    suffix = f" {DIM}[{default}]{R}" if default else ""
    val = input(f"  {CYN}{prompt}{suffix}:{R} ").strip()
    return val or default


def ensure_yt_dlp():
    if not shutil.which("yt-dlp"):
        print(f"\n  {RED}✗ yt-dlp not found on PATH.{R}")
        print(f"  {DIM}Install it with:{R} pip install -U yt-dlp\n")
        sys.exit(1)
    if not shutil.which("ffmpeg"):
        print(f"\n  {YLW}⚠ ffmpeg not found — audio conversion / video merging will fail.{R}")
        print(f"  {DIM}Install it with:{R} sudo apt install ffmpeg\n")
    if not shutil.which("node"):
        print(f"\n  {YLW}⚠ node not found — YouTube JS-challenge solving may fail.{R}")
        print(f"  {DIM}Install it with:{R} sudo apt install nodejs\n")


def is_url(text):
    return text.lower().startswith(("http://", "https://", "www."))


def fmt_duration(seconds):
    if not seconds:
        return "--:--"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ══════════════════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════════════════
def search_youtube(query, max_results=SEARCH_RESULTS):
    cmd = [
        "yt-dlp", f"ytsearch{max_results}:{query}",
        "--flat-playlist", "--dump-json", "--no-warnings",
    ]
    log(f"Searching for: {query}", DIM)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  {RED}✗ Search failed.{R}")
        if result.stderr.strip():
            print(f"  {DIM}{result.stderr.strip().splitlines()[-1]}{R}")
        return []

    hits = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        hits.append({
            "title": data.get("title") or "Untitled",
            "id": data.get("id"),
            "duration": data.get("duration"),
            "uploader": data.get("uploader") or data.get("channel") or "Unknown",
        })
    return hits


def choose_from_search(query):
    hits = search_youtube(query)
    if not hits:
        print(f"  {YLW}No results found.{R}")
        return None

    print()
    for i, h in enumerate(hits, 1):
        dur = fmt_duration(h["duration"])
        print(f"  {B}{CYN}{i:>2}{R}  {WHT}{h['title']}{R}")
        print(f"      {DIM}{h['uploader']} · {dur}{R}")
    print()

    choice = ask(f"Pick a number (1-{len(hits)}, or blank to cancel)")
    if not choice:
        return None
    if not choice.isdigit() or not (1 <= int(choice) <= len(hits)):
        print(f"  {YLW}Invalid choice.{R}")
        return None

    picked = hits[int(choice) - 1]
    print(f"  {GRN}✓ Selected: {picked['title']}{R}")
    return f"https://www.youtube.com/watch?v={picked['id']}"


# ══════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════
PLAYER_CLIENTS = ["tv", "web_safari", "ios", "android"]


def build_command(url, media_type, quality, out_dir, cookies_file, player_client=None):
    cmd = ["yt-dlp", "--js-runtimes", "node", url,
           "-o", f"{out_dir}/%(title)s.%(ext)s", "--restrict-filenames"]

    if player_client:
        cmd += ["--extractor-args", f"youtube:player_client={player_client}"]

    if cookies_file:
        cmd += ["--cookies", cookies_file]

    if media_type == "audio":
        q = AUDIO_QUALITIES.get(quality, "0")
        cmd += ["-x", "--audio-format", "mp3", "--audio-quality", q]
    else:
        fmt = VIDEO_FORMATS.get(quality, VIDEO_FORMATS["best"])
        cmd += ["-f", fmt, "--merge-output-format", "mp4"]

    return cmd


def download(url, media_type, quality, out_dir, cookies_file):
    os.makedirs(out_dir, exist_ok=True)
    attempts = [None] + PLAYER_CLIENTS

    for i, client in enumerate(attempts):
        cmd = build_command(url, media_type, quality, out_dir, cookies_file, client)
        label = client or "default"
        log(f"Attempt {i+1}/{len(attempts)} (client: {label})", DIM)
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"\n  {GRN}✓ Done — saved to {out_dir} (client: {label}){R}\n")
            return

        print(f"  {YLW}Client '{label}' failed — trying next...{R}\n")

    print(f"\n  {RED}✗ All clients failed.{R}\n")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    print(f"\n{CYN}{'═'*54}{R}")
    print(f"{B}{WHT}  YTDL v1.1 — Terminal Audio / Video Downloader{R}")
    print(f"{CYN}{'═'*54}{R}")
    print(f"  {DIM}Music  → {MUSIC_DIR}{R}")
    print(f"  {DIM}Movies → {MOVIES_DIR}{R}")
    print(f"{CYN}{'═'*54}{R}")

    ensure_yt_dlp()

    while True:
        section("New Download")

        raw = ask("URL or search term")
        if not raw:
            print(f"  {YLW}Nothing entered — try again.{R}")
            continue

        if is_url(raw):
            url = raw
        else:
            url = choose_from_search(raw)
            if not url:
                continue

        media_type = ""
        while media_type not in ("audio", "video"):
            media_type = (ask("Type (audio/video)", "audio") or "audio").lower()

        if media_type == "audio":
            print(f"  {DIM}Qualities: {', '.join(AUDIO_QUALITIES)}{R}")
            default_out = MUSIC_DIR
        else:
            print(f"  {DIM}Qualities: {', '.join(VIDEO_FORMATS)}{R}")
            default_out = MOVIES_DIR

        quality = (ask("Quality", "best") or "best").lower()
        out_dir = ask("Save to folder", default_out)

        download(url, media_type, quality, out_dir, cookies_file="")

        again = ask("Download another? (y/n)", "y")
        if again.lower() not in ("y", "yes"):
            break

    print(f"\n  {DIM}Session ended.{R}\n")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print(f"\n\n  {GRN}Catch you later!{R}\n")
