"""
Play DOOM with MCTS-based action selection.

Uses shallow MCTS rollouts (25 simulations, 20 frame depth) with the
DoomMultiVecClassifier to select actions.

Usage:
  python scripts/play_doom_mcts.py --model models/doom-multivec-trained --scenario basic
  python scripts/play_doom_mcts.py --model models/doom-multivec-trained --scenario defend_the_center \
      --simulations 50 --depth 30 --c 2.0
  python scripts/play_doom_mcts.py --model models/doom-multivec-trained --scenario basic --live
  python scripts/play_doom_mcts.py --model models/doom-multivec-trained --scenario basic \
      --batch-size 4 --simulations 100
"""

import argparse
import os
import sys
import time
from collections import Counter
from typing import List

import numpy as np
import torch
import logging

#logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import vizdoom
from doom_multivec.model.classifier import DoomMultiVecClassifier
from doom_multivec.ascii.converter import AsciiConverter
from doom_multivec.inference.mcts import MCTSAgent
from transformers import AutoTokenizer


def setup_doom(scenario='basic', visible=True, armed=False):
    """Set up VizDoom with visible window."""
    game = vizdoom.DoomGame()

    # Scenario
    scenarios = {
        'basic': vizdoom.scenarios_path + '/basic.cfg',
        'defend_the_center': vizdoom.scenarios_path + '/defend_the_center.cfg',
        'health_gathering': vizdoom.scenarios_path + '/health_gathering.cfg',
        'deadly_corridor': vizdoom.scenarios_path + '/deadly_corridor.cfg',
        'my_way_home': vizdoom.scenarios_path + '/my_way_home.cfg',
        'deathmatch': vizdoom.scenarios_path + '/deathmatch.cfg',
    }
    cfg_path = scenarios.get(scenario, scenario)
    game.load_config(cfg_path)

    # Visual settings
    game.set_window_visible(visible)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_depth_buffer_enabled(True)

    # 4 Buttons matching our actions (no strafe)
    game.clear_available_buttons()
    game.add_available_button(vizdoom.Button.ATTACK)
    game.add_available_button(vizdoom.Button.MOVE_FORWARD)
    game.add_available_button(vizdoom.Button.TURN_LEFT)
    game.add_available_button(vizdoom.Button.TURN_RIGHT)

    # Game variables
    game.add_available_game_variable(vizdoom.GameVariable.HEALTH)
    game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
    game.add_available_game_variable(vizdoom.GameVariable.KILLCOUNT)
    game.add_available_game_variable(vizdoom.GameVariable.ARMOR)

    game.set_episode_timeout(2100)  # ~60 seconds
    game.set_mode(vizdoom.Mode.PLAYER)

    game.init()
    return game


def load_model(model_path, device='cpu'):
    """Load the trained classifier. Auto-detects num_actions from saved weights."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Detect num_actions from saved state dict
    state = torch.load(os.path.join(model_path, 'model.pt'), map_location=device)
    # Find classifier weight shape or action_mlps
    for key in state:
        if 'classifier.weight' in key:
            num_actions = state[key].shape[0]
            break
        if 'attn_weight' in key:
            # attention pool mode, classifier is separate
            for k2 in state:
                if 'classifier.weight' in k2:
                    num_actions = state[k2].shape[0]
                    break
            break
    else:
        num_actions = 4  # default for human data

    model = DoomMultiVecClassifier(model_path, pool_mode='attention', num_actions=num_actions)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, tokenizer, num_actions


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_live_header():
    """Print the live mode header."""
    print("=" * 70)
    print("  MCTS DOOM - Live Mode")
    print("=" * 70)
    print()


def print_live_step_info(
    step: int,
    action_name: str,
    action_idx: int,
    health: float,
    armor: float,
    kills: float,
    reward: float,
    state_time: float,
    mcts_time: float,
    action_time: float,
    advance_time: float,
    step_time: float,
    benchmark_times: dict,
    show_timing: bool,
    action_history: List[str],
    root_children: dict,
    action_names: List[str],
    total_reward: float,
    steps_per_minute: float,
):
    """Print step information in live mode."""
    clear_screen()
    print_live_header()

    print(f"  Step: {step}")
    print(f"  Action: {action_name} (index {action_idx})")
    print()

    print("  Game State:")
    print(f"    Health: {health:.0f}")
    print(f"    Armor:  {armor:.0f}")
    print(f"    Kills:  {kills:.0f}")
    print(f"    Reward this step: {reward:.0f}")
    print(f"    Total reward: {total_reward:.0f}")
    print()

    print("  MCTS Stats:")
    print(f"    State read:    {state_time:.0f}ms")
    print(f"    Decision time: {mcts_time:.0f}ms")
    print(f"    Steps/minute: {steps_per_minute:.1f}")
    if show_timing:
        print(f"    Action exec:   {action_time:.0f}ms")
        print(f"    Tree advance:  {advance_time:.0f}ms")
        print(f"    Step total:    {step_time:.0f}ms")
        print("    Timing breakdown:")

        init_stats = benchmark_times.get("initialize_root", {})
        if init_stats:
            print(
                f"      initialize_root: {init_stats.get('total_ms', 0.0):.0f}ms total "
                f"({init_stats.get('avg_ms', 0.0):.1f}ms avg, n={int(init_stats.get('count', 0))})"
            )

        sim_stats = benchmark_times.get("run_simulation", {})
        if sim_stats:
            print(
                f"      run_simulation:  {sim_stats.get('total_ms', 0.0):.0f}ms total "
                f"({sim_stats.get('avg_ms', 0.0):.1f}ms avg, n={int(sim_stats.get('count', 0))})"
            )
            for name in ["copy_state", "select", "expand", "rollout", "backpropagate"]:
                stats = benchmark_times.get(name, {})
                if stats:
                    print(
                        f"        {name:13s}: {stats.get('total_ms', 0.0):.0f}ms total "
                        f"({stats.get('avg_ms', 0.0):.1f}ms avg, n={int(stats.get('count', 0))})"
                    )
    print()

    if root_children:
        print("  MCTS Action Distribution (visits):")
        for i, name in enumerate(action_names):
            if i in root_children:
                visits = root_children[i].visits
                value = root_children[i].total_value / max(root_children[i].visits, 1)
                bar = "█" * int(visits / 2)
                print(f"    {name:20s}: {visits:3d} visits, value={value:.2f} {bar}")
            else:
                print(f"    {name:20s}: not expanded")
        print()

    print("  Last 10 Actions (chronological):")
    if action_history:
        recent = action_history[-10:]
        for i, action in enumerate(recent, start=max(1, step - len(recent) + 1)):
            print(f"    Step {i:3d}: {action}")
    else:
        print("    No actions yet")
    print()

    print("=" * 70)


def run_episode_standard(args, game, agent, num_actions, episode, kills_start, show_timing=False):
    """Run a single episode in standard (non-live) mode.

    Returns:
        Tuple of (step_count, total_reward, action_counter, enemy_kills, latencies)
    """
    step = 0
    total_reward = 0.0
    action_counter = Counter()
    latencies = []
    simulation_times = []
    state_times = []
    action_times = []
    advance_times = []
    step_times = []
    enemy_kills = Counter()

    frame_interval = args.frame_skip / 35.0

    while not game.is_episode_finished():
        frame_start = time.perf_counter()

        state_start = time.perf_counter()
        state = game.get_state()
        if state is None:
            break
        health = game.get_game_variable(vizdoom.GameVariable.HEALTH)
        kills = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)
        state_time = (time.perf_counter() - state_start) * 1000

        # Run MCTS to select action
        t0 = time.perf_counter()
        action_name, buttons, action_idx, benchmark_times = agent.get_action()
        mcts_time = (time.perf_counter() - t0) * 1000
        latencies.append(mcts_time)
        simulation_times.append(mcts_time / args.simulations)

        # Execute action in game
        action_start = time.perf_counter()
        reward = game.make_action(buttons, args.frame_skip)
        action_time = (time.perf_counter() - action_start) * 1000
        total_reward += reward

        # Track enemy kills
        if reward > 0:
            enemy_kills[reward] += 1

        # Advance MCTS tree
        advance_start = time.perf_counter()
        agent.advance_root(action_idx)
        advance_time = (time.perf_counter() - advance_start) * 1000

        action_counter[action_name] += 1
        step += 1

        # Print stats periodically
        if step % 10 == 1:
            health = health if not game.is_episode_finished() else 0
            kills = kills if not game.is_episode_finished() else 0

            # Show tree stats
            if agent.root:
                total_visits = sum(c.visits for c in agent.root.children.values())
                visit_dist = []
                for a in range(num_actions):
                    if a in agent.root.children:
                        visits = agent.root.children[a].visits
                        pct = visits / total_visits * 100 if total_visits > 0 else 0
                        visit_dist.append(f"{MCTSAgent.ACTION_NAMES[a][:5]}={visits}({pct:.0f}%)")
            else:
                visit_dist = ["no tree"]

            print(f"  Step {step:4d} | {action_name:20s} | HP={health:.0f} K={kills:.0f} | "
                  f"{mcts_time:.0f}ms | Tree: {' '.join(visit_dist)}")
            if show_timing:
                sim_bench = benchmark_times.get("run_simulation", {})
                copy_bench = benchmark_times.get("copy_state", {})
                print(
                    f"    MCTS bench: run_simulation={sim_bench.get('total_ms', 0.0):.0f}ms "
                    f"({sim_bench.get('avg_ms', 0.0):.1f}ms avg, n={int(sim_bench.get('count', 0))})"
                )
                print(
                    f"                copy_state={copy_bench.get('total_ms', 0.0):.0f}ms "
                    f"({copy_bench.get('avg_ms', 0.0):.1f}ms avg, n={int(copy_bench.get('count', 0))})"
                )
                print(f"    Timing: state={state_time:.0f}ms mcts={mcts_time:.0f}ms action={action_time:.0f}ms advance={advance_time:.0f}ms step={(time.perf_counter() - frame_start) * 1000:.0f}ms")

        step_time = (time.perf_counter() - frame_start) * 1000
        state_times.append(state_time)
        action_times.append(action_time)
        advance_times.append(advance_time)
        step_times.append(step_time)

        # Sleep to match real-time pace
        elapsed = time.perf_counter() - frame_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

    return step, total_reward, action_counter, enemy_kills, latencies, simulation_times, state_times, action_times, advance_times, step_times


def run_episode_live(args, game, agent, num_actions, episode, show_timing=False):
    """Run a single episode in live mode.

    Returns:
        Tuple of (step_count, total_reward, action_counter, enemy_kills, latencies)
    """
    step = 0
    total_reward = 0.0
    action_history: List[str] = []
    latencies = []
    state_times = []
    action_times = []
    advance_times = []
    step_times = []
    enemy_kills = Counter()
    action_names = MCTSAgent.ACTION_NAMES[:num_actions]

    # For calculating steps per minute
    start_time = time.perf_counter()
    frame_times = []

    while not game.is_episode_finished():
        if args.steps is not None and step >= args.steps:
            print(f"\nReached max steps ({args.steps})")
            break

        step_start = time.perf_counter()

        state_start = time.perf_counter()
        state = game.get_state()
        if state is None:
            break

        health = game.get_game_variable(vizdoom.GameVariable.HEALTH)
        armor = game.get_game_variable(vizdoom.GameVariable.ARMOR)
        kills = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)
        state_time = (time.perf_counter() - state_start) * 1000

        # Run MCTS to select action
        t0 = time.perf_counter()
        action_name, buttons, action_idx, benchmark_times = agent.get_action()
        mcts_time = (time.perf_counter() - t0) * 1000
        latencies.append(mcts_time)

        # Execute action
        action_start = time.perf_counter()
        reward = game.make_action(buttons, args.frame_skip)
        action_time = (time.perf_counter() - action_start) * 1000
        total_reward += reward

        # Track enemy kills
        if reward > 0:
            enemy_kills[reward] += 1

        # Advance MCTS tree
        advance_start = time.perf_counter()
        agent.advance_root(action_idx)
        advance_time = (time.perf_counter() - advance_start) * 1000

        # Track stats
        action_history.append(action_name)
        step += 1

        # Calculate steps per minute
        frame_time = time.perf_counter() - step_start
        step_times.append(frame_time * 1000)
        state_times.append(state_time)
        action_times.append(action_time)
        advance_times.append(advance_time)
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_time_per_step = sum(frame_times) / len(frame_times) if frame_times else 1
        steps_per_minute = 60.0 / avg_time_per_step

        # Get children for display
        children = agent.root.children if agent.root else {}

        # Print step info
        print_live_step_info(
            step=step,
            action_name=action_name,
            action_idx=action_idx,
            health=health,
            armor=armor,
            kills=kills,
            reward=reward,
            state_time=state_time,
            mcts_time=mcts_time,
            action_time=action_time,
            advance_time=advance_time,
            step_time=frame_time * 1000,
            benchmark_times=benchmark_times,
            show_timing=show_timing,
            action_history=action_history,
            root_children=children,
            action_names=action_names,
            total_reward=total_reward,
            steps_per_minute=steps_per_minute,
        )

        print(f"  Status: {'Episode finished!' if game.is_episode_finished() else 'Running'}")

    return step, total_reward, Counter(action_history), enemy_kills, latencies, state_times, action_times, advance_times, step_times


def arming_sequence(game):
    """Give starting weapons and armor."""
    for _ in range(5):
        game.advance_action(1)
    game.send_game_command("give Backpack")
    for _ in range(3):
        game.advance_action(1)
    game.send_game_command("give PlasmaRifle")
    for _ in range(6):
        game.send_game_command("give CellPack")
        game.advance_action(1)
    game.send_game_command("give BlueArmor")
    game.send_game_command("use PlasmaRifle")
    game.advance_action(1)


def main():
    parser = argparse.ArgumentParser(description='Play DOOM with MCTS')
    parser.add_argument('--model', default='models/doom-multivec-trained',
                        help='Path to trained model')
    parser.add_argument('--scenario', default='defend_the_center',
                        help='DOOM scenario to play')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to play (standard mode only)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Max steps per episode (live mode)')
    parser.add_argument('--simulations', type=int, default=25,
                        help='MCTS simulations per frame')
    parser.add_argument('--depth', type=int, default=20,
                        help='Rollout depth in frames')
    parser.add_argument('--c', type=float, default=1.414,
                        help='UCB1 exploration constant (default: sqrt(2))')
    parser.add_argument('--frame-skip', type=int, default=4,
                        help='Frames between decisions')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target display FPS (standard mode only)')
    parser.add_argument('--armed', action='store_true',
                        help='Start with plasma rifle, armor, and full ammo')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--live', action='store_true',
                        help='Live mode: continuous display with steps/minute')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for parallel MCTS simulations (default: 1)')
    parser.add_argument('--use-ucb', action='store_true',
                        help='Use UCB1 instead of PUCT for tree search (default: PUCT)')
    parser.add_argument('--rollout-temperature', type=float, default=0.1,
                        help='Temperature for action sampling during rollouts (default: 0.1)')
    parser.add_argument('--prior-temperature', type=float, default=0.1,
                        help='Temperature for prior distribution modification (default: 0.1)')
    parser.add_argument('--time', action='store_true',
                        help='Print detailed MCTS timing breakdown')
    parser.add_argument('--use-llm-eval', action='store_true',
                        help='Enable LLM evaluation of rollouts')
    parser.add_argument('--llm-sampling-rate', type=int, default=4,
                        help='Frame sampling rate for LLM eval (e.g., 1 = every frame, 2 = every 2nd frame) (default: 1)')
    parser.add_argument('--llm-api-key', type=str, default=None,
                        help='API key for Triton LLM endpoint (defaults to LITELLM_API_KEY env var)')
    parser.add_argument('--llm-prompt', type=str, default=None,
                        help='Prompt for LLM evaluation')
    parser.add_argument('--llm-model', type=str, default='api-gemma-4-26b',
                        help='LLM model name to use (default: api-gemma-4-26b)')
    parser.add_argument('--llm-verbose', action='store_true',
                        help='Print LLM responses during evaluation')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print("Loading model...")
    model, tokenizer, num_actions = load_model(args.model)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Actions: {MCTSAgent.ACTION_NAMES[:num_actions]}")

    print(f"\nStarting DOOM ({args.scenario})...")
    print(f"MCTS config: {args.simulations} simulations, {args.depth} frame depth, c={args.c}, batch={args.batch_size}")
    print(f"Algorithm: {'UCB1' if args.use_ucb else 'PUCT'}, rollout_temperature={args.rollout_temperature}, prior_temperature={args.prior_temperature}")
    if args.use_llm_eval:
        print(f"LLM eval: ENABLED (sampling_rate={args.llm_sampling_rate}, model={args.llm_model})")
    if args.live:
        print("Mode: LIVE (continuous display, steps/minute)")
    if args.armed:
        print("Armed mode: Starting with plasma rifle, ammo, and armor")

    game = setup_doom(args.scenario, visible=True, armed=args.armed)
    converter = AsciiConverter(width=40, height=25)

    # Create MCTS agent
    agent = MCTSAgent(
        model=model,
        tokenizer=tokenizer,
        converter=converter,
        num_simulations=args.simulations,
        rollout_depth=args.depth,
        exploration_constant=args.c,
        num_actions=num_actions,
        device='cpu',
        batch_size=args.batch_size,
        use_puct=not args.use_ucb,
        rollout_temperature=args.rollout_temperature,
        prior_temperature=args.prior_temperature,
        use_composite_moves=True,
        composite_logit_weights=[50.0, 0.7, 0.8, 0.9],  # No additional weight on composite moves
        use_llm_eval=args.use_llm_eval,
        llm_sampling_rate=args.llm_sampling_rate,
        llm_api_key=args.llm_api_key,
        llm_prompt=args.llm_prompt,
        llm_model=args.llm_model,
        llm_verbose=args.llm_verbose,
    )
    agent.set_game(game)

    if args.live:
        # Live mode: single episode, continuous display
        game.new_episode()
        agent.reset()

        if args.armed:
            arming_sequence(game)

        step, total_reward, action_counter, enemy_kills, latencies, state_times, action_times, advance_times, step_times = run_episode_live(
            args, game, agent, num_actions, episode=1, show_timing=args.time
        )

        # Final summary
        print("\n" + "=" * 70)
        print("  FINAL SUMMARY")
        print("=" * 70)
        print(f"  Total steps: {step}")
        print(f"  Total reward: {total_reward:.0f}")
        print(f"  Final kills: {game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)}")
        print(f"  Final health: {game.get_game_variable(vizdoom.GameVariable.HEALTH)}")
        print(f"  Final armor: {game.get_game_variable(vizdoom.GameVariable.ARMOR)}")
        print("  MCTS Stats:")
        if latencies:
            print(f"    Avg decision time: {np.mean(latencies):.0f}ms")
        if args.time:
            if state_times:
                print(f"    Avg state read: {np.mean(state_times):.0f}ms")
            if action_times:
                print(f"    Avg action exec: {np.mean(action_times):.0f}ms")
            if advance_times:
                print(f"    Avg tree advance: {np.mean(advance_times):.0f}ms")
            if step_times:
                print(f"    Avg step total: {np.mean(step_times):.0f}ms")
        print()
        print("  All Actions:")
        for action, count in action_counter.most_common():
            pct = count / step * 100 if step > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {action:20s}: {count:4d} ({pct:5.1f}%) {bar}")

    else:
        # Standard mode: multiple episodes with periodic updates
        for episode in range(args.episodes):
            game.new_episode()
            agent.reset()

            if args.armed:
                arming_sequence(game)

            kills_start = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)

            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'='*60}")

            results = run_episode_standard(args, game, agent, num_actions, episode, kills_start, show_timing=args.time)
            step, total_reward, action_counter, enemy_kills, latencies, simulation_times, state_times, action_times, advance_times, step_times = results

            # Episode summary
            kills_end = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)
            kills_this_episode = int(kills_end) - int(kills_start)

            print(f"\n  --- Episode {episode + 1} Summary ---")
            print(f"  Steps: {step}")
            print(f"  Total reward: {total_reward:.0f}")
            print(f"  Kills: {kills_this_episode}")
            print("  MCTS Stats:")
            if latencies:
                print(f"    Avg decision time: {np.mean(latencies):.0f}ms")
            if args.time:
                if state_times:
                    print(f"    Avg state read: {np.mean(state_times):.0f}ms")
                if action_times:
                    print(f"    Avg action exec: {np.mean(action_times):.0f}ms")
                if advance_times:
                    print(f"    Avg tree advance: {np.mean(advance_times):.0f}ms")
                if step_times:
                    print(f"    Avg step total: {np.mean(step_times):.0f}ms")
                print(f"    Avg per simulation: {np.mean(simulation_times):.1f}ms")

            if enemy_kills:
                print(f"  Enemies killed:")
                for reward_amt in sorted(enemy_kills.keys()):
                    count = enemy_kills[reward_amt]
                    print(f"    Enemy (reward={int(reward_amt)}): {count} kills")

            print(f"  Actions:")
            for action, count in action_counter.most_common():
                pct = count / step * 100 if step > 0 else 0
                bar = '#' * int(pct / 2)
                print(f"    {action:20s}: {count:4d} ({pct:5.1f}%) {bar}")

    game.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
