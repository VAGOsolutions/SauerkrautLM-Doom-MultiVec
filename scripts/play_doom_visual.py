"""
Play DOOM visually with the trained MultiVec Classifier.
Opens a DOOM window so you can watch the model play.

Usage:
  python scripts/play_doom_visual.py --model models/doom-multivec-trained --scenario basic
  python scripts/play_doom_visual.py --model models/doom-multivec-trained --scenario defend_the_center
"""

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import vizdoom
from doom_multivec.model.classifier import DoomMultiVecClassifier
from doom_multivec.ascii.converter import AsciiConverter
from transformers import AutoTokenizer


# Map action names to vizdoom button indices
# Buttons: ATTACK, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
ACTION_TO_BUTTONS_4 = {
    'shoot':         [1, 0, 0, 0],
    'move_forward':  [0, 1, 0, 0],
    'turn_left':     [0, 0, 1, 0],
    'turn_right':    [0, 0, 0, 1],
}
# 6-action legacy (with strafe)
ACTION_TO_BUTTONS_6 = {
    'shoot':         [1, 0, 0, 0, 0, 0],
    'move_forward':  [0, 1, 0, 0, 0, 0],
    'turn_left':     [0, 0, 1, 0, 0, 0],
    'turn_right':    [0, 0, 0, 1, 0, 0],
    'strafe_left':   [0, 0, 0, 0, 1, 0],
    'strafe_right':  [0, 0, 0, 0, 0, 1],
}


def setup_doom(scenario='basic', visible=True):
    """Set up VizDoom with visible window."""
    game = vizdoom.DoomGame()

    # Scenario
    scenarios = {
        'basic': vizdoom.scenarios_path + '/basic.cfg',
        'defend_the_center': vizdoom.scenarios_path + '/defend_the_center.cfg',
        'health_gathering': vizdoom.scenarios_path + '/health_gathering.cfg',
        'deadly_corridor': vizdoom.scenarios_path + '/deadly_corridor.cfg',
        'my_way_home': vizdoom.scenarios_path + '/my_way_home.cfg',
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
    return model, tokenizer


def frame_to_action(model, tokenizer, converter, screen, depth, device='cpu',
                    top_k=2, depth_bins=None):
    """Predict action from game frame with depth.

    Returns the top action name, button vector, and all probabilities.
    """
    if depth_bins is None:
        depth_bins = getattr(model.encoder.config, 'depth_bins', 0)

    # Convert to grayscale
    if screen.ndim == 3:
        gray = np.mean(screen, axis=2).astype(np.uint8)
    else:
        gray = screen

    # ASCII + depth bins
    if depth is not None and depth_bins > 0:
        ascii_frame, depth_bins_list = converter.convert_with_depth(
            gray, depth.astype(np.float32), num_bins=depth_bins
        )
    else:
        ascii_frame = converter.convert_simple(gray)
        depth_bins_list = None

    # Tokenize
    encoded = tokenizer(
        ascii_frame,
        return_tensors='pt',
        max_length=1100,
        padding='max_length',
        truncation=True,
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Build depth_ids if available
    depth_ids = None
    if depth_bins_list is not None:
        no_depth = depth_bins
        d = [no_depth]  # CLS
        for k in range(min(len(depth_bins_list), input_ids.shape[1] - 2)):
            d.append(depth_bins_list[k])
        while len(d) < input_ids.shape[1]:
            d.append(no_depth)
        depth_ids = torch.tensor([d[:input_ids.shape[1]]], dtype=torch.long).to(device)

    with torch.no_grad():
        result = model(input_ids, attention_mask, depth_ids=depth_ids)
        probs = torch.softmax(result['logits'], dim=-1)[0].cpu().numpy()

    num_actions = len(probs)
    action_names = DoomMultiVecClassifier.ACTION_NAMES[:num_actions]
    action_map = ACTION_TO_BUTTONS_4 if num_actions <= 4 else ACTION_TO_BUTTONS_6
    sorted_idx = np.argsort(probs)[::-1]

    top_action = action_names[sorted_idx[0]]
    buttons = list(action_map[top_action])

    # Add shoot when model thinks it's relatively likely:
    # shoot if it's within 80% of the top action's probability
    shoot_idx = action_names.index('shoot') if 'shoot' in action_names else -1
    if shoot_idx >= 0 and top_action != 'shoot':
        top_prob = probs[sorted_idx[0]]
        shoot_prob = probs[shoot_idx]
        if shoot_prob > top_prob * 0.75:
            shoot_buttons = action_map['shoot']
            buttons = [max(a, b) for a, b in zip(buttons, shoot_buttons)]
            top_action = f"{top_action}+shoot"

    # Combine top-2 if compatible (movement + rotation)
    if top_k >= 2 and len(sorted_idx) > 1 and probs[sorted_idx[1]] > 0.15:
        second_action = action_names[sorted_idx[1]]
        if second_action == 'shoot':
            pass  # already handled above
        else:
            movement = {'move_forward'}
            rotation = {'turn_left', 'turn_right'}

            top_base = top_action.split('+')[0]
            top_cat = 'move' if top_base in movement else 'rot'
            sec_cat = 'move' if second_action in movement else 'rot'

            if top_cat != sec_cat:
                second_buttons = action_map[second_action]
                buttons = [max(a, b) for a, b in zip(buttons, second_buttons)]
                top_action = f"{top_action}+{second_action}"

    return top_action, buttons, probs


def main():
    parser = argparse.ArgumentParser(description='Watch DOOM MultiVec play DOOM')
    parser.add_argument('--model', default='models/doom-multivec-trained')
    parser.add_argument('--scenario', default='defend_the_center')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--frame-skip', type=int, default=4,
                        help='Frames between decisions (higher=faster, less responsive)')
    parser.add_argument('--no-composite', action='store_true',
                        help='Disable composite actions (only single action per step)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target frames per second (30=real-time, 60=fast, 10=slow-mo)')
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.model)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    print(f"\nStarting DOOM ({args.scenario})...")
    game = setup_doom(args.scenario, visible=True)
    converter = AsciiConverter(width=40, height=25)
    num_actions = len([p for p in model.parameters()])  # just need count
    # Detect from model
    for name, param in model.named_parameters():
        if 'classifier.weight' in name:
            num_actions = param.shape[0]
            break
    action_names = DoomMultiVecClassifier.ACTION_NAMES[:num_actions]
    print(f"Actions: {action_names}")
    top_k = 1 if args.no_composite else 2

    for episode in range(args.episodes):
        game.new_episode()
        step = 0
        total_reward = 0
        action_counter = Counter()
        latencies = []

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print(f"{'='*60}")

        frame_interval = args.frame_skip / 35.0

        while not game.is_episode_finished():
            frame_start = time.perf_counter()

            state = game.get_state()
            if state is None:
                break

            screen = state.screen_buffer
            depth = state.depth_buffer if hasattr(state, 'depth_buffer') else None

            t0 = time.perf_counter()
            action_name, buttons, probs = frame_to_action(
                model, tokenizer, converter, screen, depth, top_k=top_k
            )
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            reward = game.make_action(buttons, args.frame_skip)
            total_reward += reward
            action_counter[action_name] += 1
            step += 1

            # Print every 50th step
            if step % 50 == 1:
                prob_str = ' '.join(f'{action_names[i][:5]}={probs[i]:.2f}' for i in range(len(probs)))
                health = game.get_game_variable(vizdoom.GameVariable.HEALTH) if not game.is_episode_finished() else 0
                kills = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT) if not game.is_episode_finished() else 0
                print(f"  Step {step:4d} | {action_name:25s} | HP={health:.0f} K={kills:.0f} | {latency:.0f}ms | {prob_str}")

            # Sleep to match real-time pace
            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        # Episode summary
        print(f"\n  --- Episode {episode + 1} Summary ---")
        print(f"  Steps: {step}")
        print(f"  Total reward: {total_reward:.0f}")
        print(f"  Avg latency: {np.mean(latencies):.0f}ms")
        print(f"  Actions:")
        for action, count in action_counter.most_common():
            pct = count / step * 100 if step > 0 else 0
            bar = '#' * int(pct / 2)
            print(f"    {action:25s}: {count:4d} ({pct:5.1f}%) {bar}")

    game.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
