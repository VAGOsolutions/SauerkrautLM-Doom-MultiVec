"""Monte Carlo Tree Search for DOOM action selection.

Provides :class:`MCTSNode` for tree structure and :class:`MCTSAgent` for
running MCTS with the DoomMultiVecClassifier model.

The rollout function evaluates game states using:
- Win: Game reward increased during rollout
- Lose: Health decreased with no game reward increase

Batching support: run multiple leaf evaluations in parallel.
"""

import math
import os
import tempfile
import time
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path


def _save_image_index_frame(frame: Optional[np.ndarray], image_index: int) -> Optional[str]:
    """Save a node frame to `test/image{image_index}.png`.

    This is used by `_select` and `_rollout` so the captured path is
    numbered consistently across the whole simulation.
    """
    if frame is None:
        return None

    image_dir = Path(__file__).resolve().parent / "test"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"image{image_index}.png"

    array = np.asarray(frame)
    if array.ndim == 2:
        image = Image.fromarray(array.astype(np.uint8), mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    elif array.ndim == 3 and array.shape[2] == 4:
        image = Image.fromarray(array.astype(np.uint8), mode="RGBA")
    else:
        image = Image.fromarray(array.astype(np.uint8))
    image.save(image_path)
    return str(image_path)

@dataclass
class GameState:
    """Serializable game state for MCTS node storage."""
    ascii_frame: str
    depth_bins: Optional[List[int]]
    health: float
    armor: float
    game_reward: float  # Cumulative game reward at this state
    frame: Optional[np.ndarray] = None


class MCTSNode:
    """Node in the MCTS tree.

    Each node stores:
    - State representation (ASCII frame + depth bins + game vars)
    - Tree structure (parent, children)
    - MCTS statistics (visits, value)
    - Sequence of actions from root to reach this node (for state replay)

    Args:
        state: GameState containing frame and game variables
        action_sequence: List of action indices taken from root to reach this node
        parent: Parent node (None for root)
        action_taken: Action index that led to this node from parent
        num_actions: Number of possible actions
        model_priors: Optional array of action probabilities from model (for selection)
    """

    def __init__(
        self,
        state: GameState,
        action_sequence: List[int],
        parent: Optional['MCTSNode'] = None,
        action_taken: Optional[int] = None,
        num_actions: int = 4,
        model_priors: Optional[np.ndarray] = None,
    ):
        self.state = state
        self.action_sequence = action_sequence  # Actions taken from root to reach this node
        self.parent = parent
        self.action_taken = action_taken
        self.num_actions = num_actions
        self.model_priors = model_priors

        # Tree structure
        self.children: Dict[int, 'MCTSNode'] = {}
        self.untried_actions: List[int] = list(range(num_actions))

        # MCTS statistics
        self.visits = 0
        self.total_value = 0.0  # Sum of rollout outcomes
        self.prior_prob = 1.0  # From model policy (can be updated)

        # For parallel processing - is being expanded
        self.is_expanding = False

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def best_child(self, c: float = 1.414, use_puct: bool = True) -> 'MCTSNode':
        """Select best child using UCB1 or PUCT formula.

        UCB1 = Q/N + c * sqrt(log(N) / n)
        PUCT = Q/N + c * P * sqrt(N) / (1 + n)
        where P is prior probability, N is parent visits, n is child visits

        Args:
            c: Exploration constant (sqrt(2) is standard for [0,1] rewards)
            use_puct: If True, use PUCT with priors; if False, use standard UCB1

        Returns:
            Best child node according to selected formula
        """
        if not self.children:
            raise ValueError("Cannot select best child from leaf node")

        best_score = -float('inf')
        best_child = None
        

        for action, child in self.children.items():
            epsilon = 1e-5  # Small constant to prevent division by zero
            exploitation = child.total_value / (child.visits + epsilon)
            if use_puct:
                # PUCT formula with model priors
                prior = self.model_priors[action] if self.model_priors is not None else 1.0 / self.num_actions
                exploration = c * prior * math.sqrt(self.visits) / (1.0 + child.visits)
            else:
                # Standard UCB1 formula
                exploration = c * math.sqrt(math.log(self.visits) / (child.visits + epsilon))
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def best_child_parallel(self, c: float = 1.414, exclude_pending: bool = True, use_puct: bool = True) -> Optional['MCTSNode']:
        """Select best child using UCB1 or PUCT, optionally excluding nodes being expanded in parallel.

        Args:
            c: Exploration constant
            exclude_pending: If True, skip children that are currently being expanded
            use_puct: If True, use PUCT with priors; if False, use standard UCB1

        Returns:
            Best child node or None if all children are pending
        """
        if not self.children:
            raise ValueError("Cannot select best child from leaf node")

        best_score = -float('inf')
        best_child = None

        for action, child in self.children.items():
            if exclude_pending and child.is_expanding:
                continue
            epsilon = 1e-5  # Small constant to prevent division by zero
            exploitation = child.total_value / (child.visits + epsilon)
            if use_puct:
                prior = self.model_priors[action] if self.model_priors is not None else 1.0 / self.num_actions
                exploration = c * prior * math.sqrt(self.visits) / (1.0 + child.visits)
            else:
                exploration = c * math.sqrt(math.log(self.visits) / (child.visits + epsilon))
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def select_leaf_for_expansion(self, c: float = 1.414, use_puct: bool = True) -> Optional['MCTSNode']:
        """Traverse tree to find a leaf node for expansion (for parallel).

        Args:
            c: Exploration constant
            use_puct: If True, use PUCT; if False, use UCB1

        Returns:
            A leaf node ready for expansion, or None if tree is locked
        """
        node = self
        while node.is_fully_expanded() and node.children:
            node = node.best_child_parallel(c, exclude_pending=True, use_puct=use_puct)
            if node is None:
                return None  # All children expanding
        return node

    def update(self, value: float) -> None:
        """Update node statistics with a rollout result (thread-safe)."""
        self.visits += 1
        self.total_value += value

    def get_action_distribution(self) -> np.ndarray:
        """Get action probability distribution based on visit counts."""
        counts = np.zeros(self.num_actions)
        for action, child in self.children.items():
            counts[action] = child.visits

        if counts.sum() == 0:
            return np.ones(self.num_actions) / self.num_actions
        return counts / counts.sum()

    def set_root(self) -> None:
        """Convert this node to root by clearing parent reference."""
        self.parent = None
        self.action_taken = None

    def prune_siblings(self) -> None:
        """Prune all sibling subtrees to free memory.

        Called when this node becomes the new root.
        """
        if self.parent is not None:
            # Clear parent's children except self
            for action, sibling in list(self.parent.children.items()):
                if sibling is not self:
                    sibling._recursive_clear()
            self.parent.children = {self.action_taken: self}

    def _recursive_clear(self) -> None:
        """Recursively clear this subtree to free memory."""
        for child in self.children.values():
            child._recursive_clear()
        self.children.clear()
        self.parent = None
        self.action_sequence = []


class MCTSAgent:
    """MCTS agent for DOOM using DoomMultiVecClassifier.

    Performs shallow MCTS rollouts to select actions. Uses VizDoom's
    save/load to temporary files for accurate simulation.

    Supports batched simulations for parallel leaf evaluation.

    Args:
        model: DoomMultiVecClassifier for action evaluation
        tokenizer: Transformer tokenizer for ASCII frames
        converter: AsciiConverter for frame preprocessing
        num_simulations: Number of MCTS simulations per frame (default: 25)
        rollout_depth: Number of frames to simulate per rollout (default: 20)
        exploration_constant: UCB1 exploration parameter (default: sqrt(2))
        num_actions: Number of available actions (default: 4)
        device: Torch device for model inference
        temp_dir: Directory for temporary save files (default: system temp)
        batch_size: Number of parallel simulations to run (default: 1, sequential)
        use_puct: If True, use PUCT formula with model priors; if False, use standard UCB1 (default: True)
        rollout_temperature: Temperature for action sampling during rollouts. Lower=more greedy, Higher=more random (default: 1.0)
        prior_temperature: Temperature for prior distribution. Lower=sharper distribution, Higher=flatter distribution (default: 0.1)
    """

    ACTION_NAMES = ['shoot', 'move_forward', 'turn_left', 'turn_right']
    ACTION_TO_BUTTONS = {
        'shoot': [1, 0, 0, 0],
        'move_forward': [0, 1, 0, 0],
        'turn_left': [0, 0, 1, 0],
        'turn_right': [0, 0, 0, 1],
    }
    
    # Composite moves (combinations of base actions)
    COMPOSITE_MOVES = {
        'move_forward+turn_left': [0, 1, 1, 0],
        'move_forward+turn_right': [0, 1, 0, 1],
        'move_forward+shoot': [1, 1, 0, 0],
        'turn_left+shoot': [1, 0, 1, 0],
        'turn_right+shoot': [1, 0, 0, 1],
    }

    def __init__(
        self,
        model,
        tokenizer,
        converter,
        num_simulations: int = 25,
        rollout_depth: int = 20,
        exploration_constant: float = 1.414,
        num_actions: int = 4,
        device: str = 'cpu',
        temp_dir: Optional[str] = None,
        batch_size: int = 1,
        use_puct: bool = True,
        rollout_temperature: float = 0.1,
        prior_temperature: float = 0.1,
        use_composite_moves: bool = True,
        composite_logit_weights: Optional[List[float]] = None,
        save_images: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.converter = converter
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.c = exploration_constant
        self.device = device
        self.batch_size = batch_size
        self.use_puct = use_puct
        self.rollout_temperature = rollout_temperature
        self.prior_temperature = prior_temperature
        self.image_save_index = 0
        self.save_images = save_images
        self.use_composite_moves = use_composite_moves
        self.composite_logit_weights = composite_logit_weights or [1.0, 1.0, 1.0, 1.0]
        
        # Build action mappings based on composite moves toggle
        self._build_action_mappings()
        
        # Set num_actions based on actual action mappings
        self.num_actions = len(self.action_names)

        # Root node will be set when game starts
        self.root: Optional[MCTSNode] = None
        self.current_game = None
        # Store root state for replaying action sequences
        self._root_game_state: Optional[GameState] = None
        self._root_save_path: Optional[str] = None  # Path to saved root state
        self._benchmark_sink: Optional[Dict[str, Dict[str, float]]] = None

    def _record_benchmark(self, key: str, elapsed_seconds: float) -> None:
        if self._benchmark_sink is None:
            return

        entry = self._benchmark_sink.setdefault(key, {
            "total_ms": 0.0,
            "count": 0.0,
        })
        entry["total_ms"] += elapsed_seconds * 1000.0
        entry["count"] += 1.0
    
    def _build_action_mappings(self) -> None:
        """Build action name and button mappings based on composite moves setting."""
        self.action_names = list(self.ACTION_NAMES)
        self.action_to_buttons = dict(self.ACTION_TO_BUTTONS)
        
        if self.use_composite_moves:
            self.action_names.extend(self.COMPOSITE_MOVES.keys())
            self.action_to_buttons.update(self.COMPOSITE_MOVES)
            
            # Build mapping of composite action indices to base action indices
            self.composite_action_components = {
                4: [1, 2],  # move_forward+turn_left
                5: [1, 3],  # move_forward+turn_right
                6: [0, 1],  # move_forward+shoot
                7: [0, 2],  # turn_left+shoot
                8: [0, 3],  # turn_right+shoot
            }
        else:
            self.composite_action_components = {}

    def set_game(self, game) -> None:
        """Set the VizDoom game instance for state saving/loading."""
        self.current_game = game

    def reset(self) -> None:
        """Reset the tree for a new episode."""
        if self.root is not None:
            self.root._recursive_clear()
        self.root = None
        self._root_game_state = None
        # Clean up start save file
        if self._root_save_path is not None and os.path.exists(self._root_save_path):
            try:
                os.remove(self._root_save_path)
            except OSError:
                pass
        self._root_save_path = None

    def _create_state_from_game(self) -> GameState:
        """Capture current game state and save to file.

        Returns:
            Tuple of (GameState, save_file_path)
        """
        state = self.current_game.get_state()
        if state is None:
            raise RuntimeError("Game state is None - episode may have ended")
        screen = state.screen_buffer
        depth = state.depth_buffer if hasattr(state, 'depth_buffer') else None
        frame = np.array(screen, copy=True)

        # Convert to ASCII
        if screen.ndim == 3:
            gray = np.mean(screen, axis=2).astype(np.uint8)
        else:
            gray = screen

        if depth is not None:
            ascii_frame, depth_bins = self.converter.convert_with_depth(
                gray, depth.astype(np.float32), num_bins=16
            )
        else:
            ascii_frame = self.converter.convert_simple(gray)
            depth_bins = None

        # Get game variables
        health = self.current_game.get_game_variable(
            __import__('vizdoom').GameVariable.HEALTH
        )
        armor = self.current_game.get_game_variable(
            __import__('vizdoom').GameVariable.ARMOR
        )
        killcount = self.current_game.get_game_variable(
            __import__('vizdoom').GameVariable.KILLCOUNT
        )
        # Use killcount as proxy for game reward
        game_reward = float(killcount)

        game_state = GameState(
            frame=frame,
            ascii_frame=ascii_frame,
            depth_bins=depth_bins,
            health=health,
            armor=armor,
            game_reward=game_reward,
        )
        return game_state

    def _replay_action_sequence(self, action_sequence: List[int]) -> None:
        """Replay a sequence of actions from the start state.
        
        Loads the saved start state first, then replays all actions in sequence.
        All action_sequences are relative to the initial start state saved at the beginning.
        
        Args:
            action_sequence: List of action indices to replay from start
        """
        if self._root_save_path is None:
            logging.warning("_replay_action_sequence: _root_save_path not set")
            return
        
        if not os.path.exists(self._root_save_path):
            logging.warning(f"Start save file does not exist: {self._root_save_path}")
            return
        
        # Load start state
        try:
            self.current_game.load(self._root_save_path)
            logging.debug(f"Loaded start save for replay")
        except Exception:
            logging.exception(f"Failed to load start save at {self._root_save_path}")
            return
        
        # Replay each action in the sequence
        for action_idx in action_sequence:
            if self.current_game.is_episode_finished():
                break
            
            action_name = self.action_names[action_idx]
            buttons = self.action_to_buttons[action_name]
            self.current_game.make_action([0, 0, 0, 0], 1)  # Neutral action
            self.current_game.make_action(buttons, 4)  # Actual action
    
    def _load_root_state(self) -> None:
        """Load the root state into the game."""
        if self._root_game_state is None:
            logging.warning("_load_root_state: _root_game_state is None")
            return
        
        # The root state was the initial state, so we don't actually load anything
        # The game should already be at the root state after reset or initialization
        pass
    
    def _copy_state_for_expansion(self, node: MCTSNode) -> None:
        """Replay action sequence to reach node's state."""
        timing_start = time.perf_counter()
        
        try:
            # Replay actions from root to reach this node
            self._replay_action_sequence(node.action_sequence)
            logging.debug(f"Replayed {len(node.action_sequence)} actions for expansion")
        except Exception:
            logging.exception("Failed to replay action sequence for expansion")
        finally:
            self._record_benchmark("copy_state", time.perf_counter() - timing_start)

    def _save_current_frame(self, frame: Optional[np.ndarray]) -> None:
        """Save the current frame if image saving is enabled."""
        if not self.save_images:
            return

        _save_image_index_frame(frame, self.image_save_index)
        self.image_save_index += 1

    def _compute_composite_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Expand logits to include composite moves.
        
        For composite actions, the logit is computed as the weighted sum of the subcomponents'
        logits, allowing a more expressive probability distribution beyond the base 4 actions.
        
        Args:
            logits: Tensor of shape (4,) with base action logits
            
        Returns:
            Tensor of shape (9,) with base + composite action logits (if composite moves enabled)
            Otherwise returns original logits padded/returned as-is
        """
        if not self.use_composite_moves:
            return logits
        
        # Start with base action logits
        expanded_logits = logits.clone() 
        
        # Compute composite action logits by applying weights and summing component logits
        for composite_idx, component_indices in self.composite_action_components.items():
            composite_logit = sum(self.composite_logit_weights[i] * logits[i] for i in component_indices)
            expanded_logits = torch.cat([expanded_logits, composite_logit.unsqueeze(0)])
        
        return expanded_logits

    def _evaluate_state(self, input_ids, attention_mask, depth_ids) -> np.ndarray:
        """Get action probabilities from the model.

        Applies prior_temperature to sharpen or flatten the distribution.
        Temperature < 1.0 sharpens the distribution (more peaked).
        Temperature > 1.0 flattens the distribution (more uniform).
        
        If composite moves are enabled, expands the distribution to include
        composite actions computed from base action logits.

        Returns:
            Array of action probabilities
        """
        with torch.no_grad():
            result = self.model(input_ids, attention_mask, depth_ids=depth_ids)
            logits = result['logits'][0]
            
            # Expand logits to include composite moves if enabled
            logits = self._compute_composite_logits(logits)

            #print dict of logits and corresponding action names for debugging
            logit_dict = {self.action_names[i]: logits[i].item() for i in range(len(self.action_names))}
            # logging.debug(f"Model logits: {logit_dict}")
            
            # Apply prior_temperature scaling
            if self.prior_temperature != 1.0:
                logits = logits / self.prior_temperature
            
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs[:self.num_actions]

    def _prepare_model_input(self, state: GameState):
        """Prepare model input from game state."""
        encoded = self.tokenizer(
            state.ascii_frame,
            return_tensors='pt',
            max_length=1100,
            padding='max_length',
            truncation=True,
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Build depth_ids if available
        depth_ids = None
        if state.depth_bins is not None:
            no_depth = 16
            d = [no_depth]  # CLS
            d.extend(state.depth_bins[:input_ids.shape[1] - 2])
            while len(d) < input_ids.shape[1]:
                d.append(no_depth)
            depth_ids = torch.tensor([d[:input_ids.shape[1]]], dtype=torch.long).to(self.device)

        return input_ids, attention_mask, depth_ids

    def _sample_action_from_current_state(self) -> Tuple[int, np.ndarray]:
        """Get current game state, evaluate model, and sample an action.

        Returns:
            Tuple of (sampled action index, action probability distribution)
        """
        state_obj = self.current_game.get_state()
        if state_obj is None:
            raw_action_probs = np.ones(self.num_actions) / self.num_actions
            return np.random.randint(self.num_actions), raw_action_probs

        screen = state_obj.screen_buffer
        depth = state_obj.depth_buffer if hasattr(state_obj, 'depth_buffer') else None

        # Convert to ASCII
        if screen.ndim == 3:
            gray = np.mean(screen, axis=2).astype(np.uint8)
        else:
            gray = screen

        if depth is not None:
            ascii_frame, depth_bins = self.converter.convert_with_depth(
                gray, depth.astype(np.float32), num_bins=16
            )
        else:
            ascii_frame = self.converter.convert_simple(gray)
            depth_bins = None

        # Create temporary state for model evaluation
        rollout_state = GameState(
            frame=np.array(screen, copy=True),
            ascii_frame=ascii_frame,
            depth_bins=depth_bins,
            health=self.current_game.get_game_variable(__import__('vizdoom').GameVariable.HEALTH),
            armor=self.current_game.get_game_variable(__import__('vizdoom').GameVariable.ARMOR),
            game_reward=float(self.current_game.get_game_variable(__import__('vizdoom').GameVariable.KILLCOUNT)),
        )

        # Get model priors for action selection
        input_ids, attention_mask, depth_ids = self._prepare_model_input(rollout_state)
        raw_action_probs = self._evaluate_state(input_ids, attention_mask, depth_ids)

        # Apply temperature scaling to control exploration during rollout
        action_probs = raw_action_probs.copy()
        if self.rollout_temperature != 1.0:
            action_probs = np.power(action_probs, 1.0 / self.rollout_temperature)
            action_probs = action_probs / action_probs.sum()

        # Sample action from model distribution
        action = np.random.choice(self.num_actions, p=action_probs)
        return action, raw_action_probs

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection: traverse tree using UCB1 or PUCT until reaching unexpanded node."""
        timing_start = time.perf_counter()
        self._save_current_frame(node.state.frame)
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c, use_puct=self.use_puct)
            self._save_current_frame(node.state.frame)
        self._record_benchmark("select", time.perf_counter() - timing_start)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion: create child node for an untried action.
        Uses model priors from parent node to bias action selection.
        """
        timing_start = time.perf_counter()
        if not node.untried_actions:
            self._record_benchmark("expand", time.perf_counter() - timing_start)
            return node

        # Select untried action, preferring high-prior actions
        if node.model_priors is not None:
            # Choose action with highest prior among untried actions
            best_action = max(node.untried_actions, key=lambda a: node.model_priors[a])
            node.untried_actions.remove(best_action)
            action = best_action
        else:
            # Fallback to sequential selection
            action = node.untried_actions.pop(0)

        # Replay to parent state and take action
        self._copy_state_for_expansion(node)
        action_name = self.action_names[action]
        buttons = self.action_to_buttons[action_name]
        self.current_game.make_action([0,0,0,0], 1) 
        self.current_game.make_action(buttons, 4)
        
        rollout_state = self.current_game.get_state()
        if rollout_state is not None:
            self._save_current_frame(rollout_state.screen_buffer)

        # Capture new state
        child_state = self._create_state_from_game()

        # Evaluate model priors for child state (will be used when selecting its children)
        input_ids, attention_mask, depth_ids = self._prepare_model_input(child_state)
        child_priors = self._evaluate_state(input_ids, attention_mask, depth_ids)

        logging.debug(f"Expand action: {action_name}")
        # Create child node with action sequence
        child_action_sequence = node.action_sequence + [action]
        child = MCTSNode(
            state=child_state,
            action_sequence=child_action_sequence,
            parent=node,
            action_taken=action,
            num_actions=self.num_actions,
            model_priors=child_priors,
        )

        node.children[action] = child
        self._record_benchmark("expand", time.perf_counter() - timing_start)
        return child

    def _rollout(self, node: MCTSNode) -> float:
        """Rollout: simulate actions using model priors and return binary win/lose.

        Win: Game reward increased during rollout
        Lose: Health OR armor decreased with no game reward increase
        
        Assumes the game is already positioned at node's state (from prior _expand call).

        Returns:
            1.0 for win, 0.0 for lose, 0.5 for neutral
        """
        timing_start = time.perf_counter()

        # Record starting metrics
        vizdoom = __import__('vizdoom')
        start_health = self.current_game.get_game_variable(
            vizdoom.GameVariable.HEALTH
        )
        start_armor = self.current_game.get_game_variable(
            vizdoom.GameVariable.ARMOR
        )
        start_kills = self.current_game.get_game_variable(
            vizdoom.GameVariable.KILLCOUNT
        )

        # Simulate actions guided by model
        for _ in range(self.rollout_depth):
            if self.current_game.is_episode_finished():
                break

            # Get action from model and execute it
            action, raw_action_probs = self._sample_action_from_current_state()
            logging.debug(f"Rollout action: {self.action_names[action]}")
            action_name = self.action_names[action]
            buttons = self.action_to_buttons[action_name]
            self.current_game.make_action(buttons, 4)

            rollout_state = self.current_game.get_state()
            if rollout_state is not None:
                self._save_current_frame(rollout_state.screen_buffer)
            
        # Evaluate outcome
        end_health = self.current_game.get_game_variable(
            vizdoom.GameVariable.HEALTH
        )
        end_armor = self.current_game.get_game_variable(
            vizdoom.GameVariable.ARMOR
        )
        end_kills = self.current_game.get_game_variable(
            vizdoom.GameVariable.KILLCOUNT
        )

        kill_reward = np.sign(end_kills - start_kills)
        health_reward = np.sign(end_health - start_health)
        armor_reward = np.sign(end_armor - start_armor)

        value = kill_reward + 2 * health_reward + 2 * armor_reward

        self._record_benchmark("rollout", time.perf_counter() - timing_start)
        return value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagation: update statistics up the tree."""
        timing_start = time.perf_counter()
        current = node
        while current is not None:
            current.update(value)
            current = current.parent
        self._record_benchmark("backpropagate", time.perf_counter() - timing_start)

    def run_simulation(self) -> None:
        """Run one MCTS simulation (select, expand, rollout, backprop)."""
        timing_start = time.perf_counter()
        if self.root is None:
            raise ValueError("Root node not initialized. Call initialize_root() first.")

        # Always leave simulation with root state loaded so rollout actions do
        # not leak into live gameplay.
        try:
            if self.save_images:
                self.image_save_index = 0

            # Selection
            node = self._select(self.root)

            # Expansion (if not terminal)
            if not self.current_game.is_episode_finished() and node.untried_actions:
                node = self._expand(node)

            # Rollout (only if game still running)
            if self.current_game.is_episode_finished():
                value = 0.5  # Episode ended
            else:
                value = self._rollout(node)

            # Backpropagation
            self._backpropagate(node, value)
        finally:
            self._record_benchmark("run_simulation", time.perf_counter() - timing_start)

    def _run_single_leaf_eval(self, leaf_info: Tuple[MCTSNode, str, int]) -> Tuple[MCTSNode, float]:
        """Run a single leaf evaluation for batching.

        Args:
            leaf_info: Tuple of (node, action_sequence, action_to_expand)

        Returns:
            Tuple of (node, value)
        """
        node, action_sequence, action = leaf_info

        # Replay action sequence to reach state
        self._replay_action_sequence(action_sequence)

        # Take action
        action_name = self.action_names[action]
        buttons = self.action_to_buttons[action_name]
        self.current_game.make_action(buttons, 4)

        # Capture new state
        child_state = self._create_state_from_game()

        # Evaluate model priors for child state (will be used when selecting its children)
        input_ids, attention_mask, depth_ids = self._prepare_model_input(child_state)
        child_priors = self._evaluate_state(input_ids, attention_mask, depth_ids)

        # Create child node with action sequence
        child_action_sequence = action_sequence + [action]
        child = MCTSNode(
            state=child_state,
            action_sequence=child_action_sequence,
            parent=node,
            action_taken=action,
            num_actions=self.num_actions,
            model_priors=child_priors,
        )
        node.children[action] = child

        # Rollout
        if self.current_game.is_episode_finished():
            value = 0.5
        else:
            value = self._rollout(child)

        return child, value

    def run_simulations_batched(self) -> None:
        """Run multiple MCTS simulations with batching support.

        This runs num_simulations MCTS simulations, batching the leaf
        evaluations for parallel processing. Note: True parallelization
        requires multiple game instances; this version runs sequentially
        but is structured for easy parallel extension.
        """
        if self.root is None:
            raise ValueError("Root node not initialized. Call initialize_root() first.")

        # For now, run simulations sequentially
        # Can be extended to parallel with ThreadPoolExecutor if using multiple game instances
        for _ in range(self.num_simulations):
            self.run_simulation()

    def initialize_root(self) -> None:
        """Create root node from current game state with model priors."""
        timing_start = time.perf_counter()
        game_state = self._create_state_from_game()
        self._root_game_state = game_state  # Store root state for reference
        
        # Save start state to file for replaying action sequences (only once)
        if self._root_save_path is None:
            self._root_save_path = os.path.join(tempfile.gettempdir(), f'mcts_start_save_{id(self)}.zds')
            try:
                self.current_game.save(self._root_save_path)
                logging.debug(f"Saved start state to {self._root_save_path}")
            except Exception:
                logging.exception(f"Failed to save start state to {self._root_save_path}")

        # Evaluate model priors for root state
        input_ids, attention_mask, depth_ids = self._prepare_model_input(game_state)
        model_priors = self._evaluate_state(input_ids, attention_mask, depth_ids)

        self.root = MCTSNode(
            state=game_state,
            action_sequence=[],  # Root has no actions
            parent=None,
            action_taken=None,
            num_actions=self.num_actions,
            model_priors=model_priors,
        )
        self._record_benchmark("initialize_root", time.perf_counter() - timing_start)

    def get_action(self) -> Tuple[str, List[int], int, Dict[str, Dict[str, float]]]:
        """Run MCTS and return the best action.

        Returns:
            Tuple of (action_name, button_vector, action_index, benchmark_times)
        """
        benchmark_times: Dict[str, Dict[str, float]] = {
            "initialize_root": {"total_ms": 0.0, "count": 0.0},
            "run_simulation": {"total_ms": 0.0, "count": 0.0},
            "copy_state": {"total_ms": 0.0, "count": 0.0},
            "select": {"total_ms": 0.0, "count": 0.0},
            "expand": {"total_ms": 0.0, "count": 0.0},
            "rollout": {"total_ms": 0.0, "count": 0.0},
            "backpropagate": {"total_ms": 0.0, "count": 0.0},
        }
        self._benchmark_sink = benchmark_times

        try:
            if self.root is None:
                self.initialize_root()

            # Run simulations (batched if batch_size > 1)
            if self.batch_size > 1:
                self.run_simulations_batched()
            else:
                for _ in range(self.num_simulations):
                    self.run_simulation()

            # Select best action (most visits)
            best_action = 0
            best_visits = -1
            for action, child in self.root.children.items():
                if child.visits > best_visits:
                    best_visits = child.visits
                    best_action = action

            action_name = self.action_names[best_action]
            buttons = self.action_to_buttons[action_name]

            # Ensure caller executes chosen action from real/root game state.
            self._copy_state_for_expansion(self.root)

            for entry in benchmark_times.values():
                count = int(entry["count"])
                entry["avg_ms"] = entry["total_ms"] / count if count else 0.0

            return action_name, buttons, best_action, benchmark_times
        finally:
            self._benchmark_sink = None

    def advance_root(self, action_taken: int) -> None:
        """Advance tree by making the selected action the new root.

        This prunes all sibling branches and promotes the selected child.
        """
        if self.root is None:
            return

        if self.current_game.is_episode_finished():
            # Episode ended, clear tree
            self.reset()
            return

        if action_taken not in self.root.children:
            # Action not in tree, create new root from current state
            self.initialize_root()
            return

        # Get the child that becomes new root
        new_root = self.root.children[action_taken]

        # Prune siblings
        new_root.prune_siblings()

        # Detach from parent
        new_root.set_root()

        # Update root reference
        self.root = new_root

        # Update state from current game position (if game still running)
        if not self.current_game.is_episode_finished():
            try:
                game_state = self._create_state_from_game()
                self.root.state = game_state
                self._root_game_state = game_state
            except RuntimeError:
                # State not available, keep existing
                pass
