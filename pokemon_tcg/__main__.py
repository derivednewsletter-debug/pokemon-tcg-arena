"""Pokemon TCG AI Battle Challenge — CLI.

Usage:
    python3 -m pokemon_tcg tests
    python3 -m pokemon_tcg simulate --seed 42 --turns 40
    python3 -m pokemon_tcg benchmark --agent Greedy --seed 42
    python3 -m pokemon_tcg tournament --seeds 42,7,123
    python3 -m pokemon_tcg leaderboard --results results/

The CLI mirrors kaggriculture's main.py so users can swap between the
two projects without re-learning the workflow.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def cmd_tests(args):
    from tests import run as _run
    sys.exit(_run(filter=args.filter))


def cmd_simulate(args):
    from .agents.benchmarks import BENCHMARKS
    from .simulator import simulate_match
    from .experiments.deck_pool import build_deck_pool

    decks = build_deck_pool("random", n=2)
    a, b = (BENCHMARKS[args.a](), BENCHMARKS[args.b]()) if args.a == args.b else \
           (BENCHMARKS[args.a](), BENCHMARKS[args.b]())
    result = simulate_match(decks.decks[0], decks.decks[1], [a, b],
                             seed=args.seed, log=False, max_turns=args.turns)
    print(f"seed={args.seed}  winner={result['winner']}  turns={result['turns']}")
    print("\nEvents:")
    for e in result["log"][:80]:
        print(" ", json.dumps(e, default=str))


def cmd_benchmark(args):
    from .agents.benchmarks import BENCHMARKS
    from .experiments.runner import ExperimentRunner, run_batch
    from .experiments.deck_pool import build_deck_pool

    seeds = [int(x) for x in args.seeds.split(",")]
    opponents = [BENCHMARKS[n]() for n in (args.opponents or "Greedy,Defensive,Aggressive").split(",")]
    results = run_batch(BENCHMARKS[args.agent](), opponents, seeds,
                        out_dir=args.out, games_per_pair=args.games)
    for r in results:
        m = r.metrics.to_dict()
        print(f"\n=== {r.agent} vs {r.opponent} ===")
        print(json.dumps(m, indent=2))
        save_path = os.path.join(args.out, f"benchmark_{r.agent}_vs_{r.opponent}.json")
        r.save(save_path)
        print(f"-> saved {save_path}")


def cmd_tournament(args):
    from .agents.benchmarks import BENCHMARKS
    from .experiments.tournament import run_tournament
    seeds = [int(x) for x in args.seeds.split(",")]
    names = (args.agents or "Champion,SearchAgent,Greedy,Aggressive,Defensive,BenchBuffer").split(",")
    agents = []
    available = set(BENCHMARKS.keys())
    for n in names:
        if n not in available:
            print(f"warning: agent '{n}' not registered; using Greedy")
            n = "Greedy"
        agents.append(BENCHMARKS[n]())
    res = run_tournament(agents, seeds, pool_mode=args.pool,
                          games_per_pair=args.games, out_dir=args.out)
    print(res.markdown())
    if args.save:
        with open(args.save, "w") as fh:
            fh.write(json.dumps({
                "elo": res.leaderboard.elo,
                "matchup": {f"{a}_vs_{b}": list(v)
                            for (a, b), v in res.leaderboard.matchup_wins.items()},
                "records": [r.to_json() for r in res.records],
            }, default=str))
        print(f"\nsaved -> {args.save}")


def cmd_leaderboard(args):
    print("(leaderboard aggregation not yet implemented; use tournament save)")


def cmd_analyze(args):
    from .analysis.failure import summarize_failures
    from .logging_utils.game_log import MatchRecord, read_match_json
    print(f"reading {args.path} ... ")
    if args.path.endswith(".jsonl"):
        records = []
        with open(args.path) as fh:
            for line in fh:
                if line.strip():
                    records.append(MatchRecord.from_json(json.loads(line)))
    else:
        records = [read_match_json(args.path)]
    fa = summarize_failures(records, perspective=args.perspective,
                             opponent_name=args.opponent)
    print(json.dumps(fa.totals(), indent=2))
    print("dominant:", fa.dominant_failure())


def cmd_export_decks(args):
    from .experiments.deck_pool import build_deck_pool
    pool = build_deck_pool(args.mode, n=args.n, base_seed=args.seed)
    print(f"built {len(pool)} decks (mode='{args.mode}', seed={args.seed})")


def cmd_train_value(args):
    """Run self-play + train a small MLP value head and save weights.

    The default spawn is Champion vs Greedy which already gives a
    meaningful gradient (90% of examples are non-zero targets).
    """
    from .selfplay import (
        collect_examples, train, save_model, run_selfplay_and_train,
    )

    spawn_a = args.spawn_a
    spawn_b = args.spawn_b
    if spawn_a == "Champion":
        from .agents.champion import ChampionAgent as a
    elif spawn_a == "Greedy":
        from .agents.benchmarks import GreedyAgent as a
    elif spawn_a == "SearchAgent":
        from .agents.benchmarks import SearchAgent as a
    else:
        raise SystemExit(f"unknown spawn_a '{spawn_a}'")
    if spawn_b == "Champion":
        from .agents.champion import ChampionAgent as b
    elif spawn_b == "Greedy":
        from .agents.benchmarks import GreedyAgent as b
    elif spawn_b == "SearchAgent":
        from .agents.benchmarks import SearchAgent as b
    else:
        raise SystemExit(f"unknown spawn_b '{spawn_b}'")

    if args.fast:
        # Quick path: short, single-shot training run
        run_selfplay_and_train(
            num_games=args.games,
            spawn_a=a, spawn_b=b,
            epochs=args.epochs, lr=args.lr,
            out_path=args.out,
        )
        return

    # Standard path with explicit diagnostics
    sp = collect_examples(a, b, num_games=args.games)
    print(f"[selfplay] games={sp.games_played} examples={len(sp.examples)} "
          f"(wins_p0={sp.wins_p0}, wins_p1={sp.wins_p1}, draws={sp.draws})")
    print(f"[selfplay] elapsed={sp.elapsed_sec:.1f}s")
    import numpy as np
    x, y = sp.to_arrays()
    print(f"[features] X shape={x.shape} Y shape={y.shape}")
    print(f"[targets]  distribution={dict(zip(*np.unique(y, return_counts=True)))}")
    model = train(sp, epochs=args.epochs, lr=args.lr, verbose=True)
    save_model(model, args.out)
    print(f"[save] -> {args.out}")


def cmd_train_retreat_threshold(args):
    """Train a logistic-regression retreat predictor and recommend a threshold.

    Runs Champion-vs-Greedy self-play by default, snapshots the
    retreat-relevant features + eventual game outcome, fits a small
    logistic regression, and picks the per-turn probability threshold
    that maximises accuracy on a held-out quarter of the data.

    Outputs
    -------
    * ``--out`` (``results/retreat_model.npz``): the trained weights.
    * ``--threshold-out`` (``results/retreat_threshold.json``): the
      recommended threshold + per-threshold accuracy grid.
    """
    from .analysis.retreat_model import train_and_pick_threshold

    spawn_map = {
        "Champion": "pokemon_tcg.agents.champion:ChampionAgent",
        "Greedy": "pokemon_tcg.agents.benchmarks:GreedyAgent",
        "SearchAgent": "pokemon_tcg.agents.benchmarks:SearchAgent",
    }
    if args.spawn_a not in spawn_map:
        raise SystemExit(f"unknown --spawn-a '{args.spawn_a}' (valid: "
                         f"{', '.join(spawn_map)})")
    if args.spawn_b not in spawn_map:
        raise SystemExit(f"unknown --spawn-b '{args.spawn_b}' (valid: "
                         f"{', '.join(spawn_map)})")
    import importlib
    mod_a, name_a = spawn_map[args.spawn_a].split(":")
    mod_b, name_b = spawn_map[args.spawn_b].split(":")
    spawn_a = getattr(importlib.import_module(mod_a), name_a)
    spawn_b = getattr(importlib.import_module(mod_b), name_b)

    predictor = train_and_pick_threshold(
        num_games=args.games,
        spawn_a=spawn_a, spawn_b=spawn_b,
        epochs=args.epochs, lr=args.lr,
        out_path=args.out, threshold_out=args.threshold_out,
        verbose=True,
    )
    print()
    print(f"[retreat] predictor: {predictor!r}")
    print(f"[retreat] weights -> {args.out}")
    print(f"[retreat] threshold -> {args.threshold_out}")
    print()
    print("To use this predictor in Champion, pass it via:")
    print(f'  ChampionAgent(config={{"retreat_predictor": '
          f'RetreatWinPredictor.load("{args.out}")}})')


def cmd_web(args):
    from .webapp.app import create_app
    print(f"Launching Pokemon TCG dashboard on http://{args.host}:{args.port}")
    print("Routes:")
    print("  /           Director dashboard (Elo leaderboard + win-rate cards)")
    print("  /matrix     Matchup-matrix heatmap")
    print("  /replay     Step through a match turn by turn")
    create_app().run(host=args.host, port=args.port, debug=False)


def main():
    p = argparse.ArgumentParser(prog="pokemon_tcg")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tests")
    t.add_argument("--filter", default=None)

    s = sub.add_parser("simulate")
    s.add_argument("--a", default="Greedy")
    s.add_argument("--b", default="Defensive")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--turns", type=int, default=40)

    b = sub.add_parser("benchmark")
    b.add_argument("--agent", default="Greedy")
    b.add_argument("--opponents", default=None)
    b.add_argument("--seeds", default="42,7,123")
    b.add_argument("--games", type=int, default=2)
    b.add_argument("--out", default="results")

    to = sub.add_parser("tournament")
    to.add_argument("--agents", default=None)
    to.add_argument("--seeds", default="42,7,123")
    to.add_argument("--games", type=int, default=2)
    to.add_argument("--pool", default="themed")
    to.add_argument("--out", default="results")
    to.add_argument("--save", default="results/tournament.json")

    sub.add_parser("leaderboard")
    a = sub.add_parser("analyze")
    a.add_argument("path")
    a.add_argument("--perspective", type=int, default=0)
    a.add_argument("--opponent", default="?")

    e = sub.add_parser("export-decks")
    e.add_argument("--mode", default="themed")
    e.add_argument("--n", type=int, default=8)
    e.add_argument("--seed", type=int, default=42)

    w = sub.add_parser("web", help="launch the dashboard")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=5055)

    t = sub.add_parser("train-value",
                        help="Run self-play + train the MLP value head.")
    t.add_argument("--games", type=int, default=8,
                   help="number of self-play games to collect (default 8)")
    t.add_argument("--epochs", type=int, default=20,
                   help="SGD epochs over the collected examples")
    t.add_argument("--lr", type=float, default=0.01,
                   help="learning rate (default 0.01)")
    t.add_argument("--spawn-a", default="Champion",
                   help="player-0 factory: Champion / Greedy / SearchAgent "
                        "(default Champion)")
    t.add_argument("--spawn-b", default="Greedy",
                   help="player-1 factory: Champion / Greedy / SearchAgent "
                        "(default Greedy)")
    t.add_argument("--out", default="results/value_head.npz",
                   help="where to save the trained MLP weights")
    t.add_argument("--fast", action="store_true",
                   help="shortcut: minimal logging, 8 games / 20 epochs")

    r = sub.add_parser("train-retreat-threshold",
                        help="Train a logistic-regression retreat predictor "
                             "and emit a recommended retreat threshold.")
    r.add_argument("--games", type=int, default=16,
                   help="number of self-play games to harvest (default 16)")
    r.add_argument("--epochs", type=int, default=60,
                   help="SGD epochs (default 60)")
    r.add_argument("--lr", type=float, default=0.05,
                   help="learning rate (default 0.05)")
    r.add_argument("--spawn-a", default="Champion",
                   help="player-0 factory (Champion / Greedy / SearchAgent)")
    r.add_argument("--spawn-b", default="Greedy",
                   help="player-1 factory (Champion / Greedy / SearchAgent)")
    r.add_argument("--out", default="results/retreat_model.npz",
                   help="where to save the trained logistic weights")
    r.add_argument("--threshold-out",
                   default="results/retreat_threshold.json",
                   help="where to save the recommended threshold JSON")

    args = p.parse_args()
    {"tests": cmd_tests, "simulate": cmd_simulate, "benchmark": cmd_benchmark,
     "tournament": cmd_tournament, "leaderboard": cmd_leaderboard,
     "analyze": cmd_analyze, "export-decks": cmd_export_decks,
     "web": cmd_web, "train-value": cmd_train_value,
     "train-retreat-threshold": cmd_train_retreat_threshold,
     }[args.cmd](args)


if __name__ == "__main__":
    main()
