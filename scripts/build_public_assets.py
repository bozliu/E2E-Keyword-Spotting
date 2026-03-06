#!/usr/bin/env python
"""Build sanitized public figures for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from kws.demo.visuals import apply_theme, build_wheel, create_hud, resolve_active_index, update_wheel

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'assets'
BENCH = ASSETS / 'benchmarks'
DEMO = ASSETS / 'demo'
FIGS = ASSETS / 'figures'


def _ensure_dirs() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def build_demo_figure() -> None:
    labels = ['silence', 'unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    fig, ax = plt.subplots(figsize=(9, 10), dpi=180)
    apply_theme(fig, ax)
    hud = create_hud(fig, ax, title_text='Google Speech Dataset Demo')
    wheel = build_wheel(ax, labels, radius=1.05, fontsize=12, ring_width=0.42, place_labels='sector_center')
    update_wheel(wheel, resolve_active_index(labels, 'go'), explode=0.06)
    hud.prompt.set_text('[LISTENING]')
    hud.center.set_text('GO')
    hud.center.set_fontsize(34)
    hud.status.set_text('Latest public demo visualized from the release assets')
    fig.tight_layout(pad=0.2)
    fig.savefig(DEMO / 'google_speech_demo.png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def build_focus_figure() -> None:
    df = pd.read_csv(BENCH / 'focus_confusions.csv')
    runs = ['demo_mhatt_small_focus', 'demo_mhatt_small_focus_lod']
    labels = {
        'demo_mhatt_small_focus': 'Before LOD refinement',
        'demo_mhatt_small_focus_lod': 'Latest public demo',
    }
    keywords = ['left', 'on', 'down']
    width = 0.36
    x = range(len(keywords))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    fig.patch.set_facecolor('white')
    colors = ['#7f8cff', '#1f295a']
    for idx, run in enumerate(runs):
        rates = [float(df[(df['run'] == run) & (df['keyword'] == kw)]['confusion_rate'].iloc[0]) * 100.0 for kw in keywords]
        xpos = [i + (idx - 0.5) * width for i in x]
        bars = ax.bar(xpos, rates, width=width, label=labels[run], color=colors[idx])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([kw.upper() for kw in keywords], fontsize=11)
    ax.set_ylabel('Focused confusion rate (%)')
    ax.set_title('Targeted confusion reduction on hard keywords')
    ax.legend(frameon=False)
    ax.set_ylim(0, max(ax.get_ylim()[1], 20))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / 'focus_confusion_reduction.png', bbox_inches='tight')
    plt.close(fig)


def build_latency_tradeoff() -> None:
    df = pd.read_csv(BENCH / 'demo_benchmark.csv')
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    fig.patch.set_facecolor('white')
    colors = {
        'quick_mhatt': '#7a7f8f',
        'demo_mhatt_small_focus': '#6d77ff',
        'demo_mhatt_small_focus_lod': '#131d4f',
    }
    for _, row in df.iterrows():
        run = row['run']
        latency = float(row['latency_ms'])
        recall = float(row['kws12_target_recall']) * 100.0
        size = float(row['params']) / 5000.0
        ax.scatter(latency, recall, s=size, color=colors.get(run, '#4c4c4c'), alpha=0.9)
        ax.text(latency + 0.15, recall + 0.15, row['display_name'], fontsize=9)
    ax.set_xlabel('Median CPU latency (ms)')
    ax.set_ylabel('KWS12 target recall (%)')
    ax.set_title('Realtime deployment trade-off: latency vs. recall')
    ax.grid(alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / 'latency_vs_recall.png', bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    _ensure_dirs()
    build_demo_figure()
    build_focus_figure()
    build_latency_tradeoff()
    print('Built public assets in', ASSETS)


if __name__ == '__main__':
    main()
