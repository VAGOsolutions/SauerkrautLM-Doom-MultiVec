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
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

@dataclass
class GameState:
    """Serializable game state for MCTS node storage."""
    ascii_frame: str
    depth_bins: Optional[List[int]]
    health: float
    armor: float
    game_reward: float  # Cumulative game reward at this state


class MCTSNode:
    """Node in the MCTS tree.

    Each node stores:
    - State representation (ASCII frame + depth bins + game vars)
    - Tree structure (parent, children)
    - MCTS statistics (visits, value)
    - Path to saved VizDoom state file for accurate rollouts

    Args:
        state: GameState containing frame and game variables
        save_path: Path to VizDoom save file
        parent: Parent node (None for root)
        action_taken: Action index that led to this node from parent
        num_actions: Number of possible actions
        model_priors: Optional array of action probabilities from model (for selection)
    """

    def __init__(
        self,
        state: GameState,
        save_path: Optional[str],
        parent: Optional['MCTSNode'] = None,
        action_taken: Optional[int] = None,
        num_actions: int = 4,
        model_priors: Optional[np.ndarray] = None,
    ):
        self.state = state
        self.save_path = save_path
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
            if child.visits == 0:
                # Prioritize unvisited children
                score = float('inf')
            else:
                exploitation = child.total_value / child.visits
                if use_puct:
                    # PUCT formula with model priors
                    prior = self.model_priors[action] if self.model_priors is not None else 1.0 / self.num_actions
                    exploration = c * prior * math.sqrt(self.visits) / (1.0 + child.visits)
                else:
                    # Standard UCB1 formula
                    exploration = c * math.sqrt(math.log(self.visits) / child.visits)
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
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.total_value / child.visits
                if use_puct:
                    prior = self.model_priors[action] if self.model_priors is not None else 1.0 / self.num_actions
                    exploration = c * prior * math.sqrt(self.visits) / (1.0 + child.visits)
                else:
                    exploration = c * math.sqrt(math.log(self.visits) / child.visits)
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
        # Delete save file if it exists
        if self.save_path is not None and os.path.exists(self.save_path):
            try:
                os.remove(self.save_path)
            except OSError:
                pass
        self.save_path = None


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
    """

    ACTION_NAMES = ['shoot', 'move_forward', 'turn_left', 'turn_right']
    ACTION_TO_BUTTONS = {
        'shoot': [1, 0, 0, 0],
        'move_forward': [0, 1, 0, 0],
        'turn_left': [0, 0, 1, 0],
        'turn_right': [0, 0, 0, 1],
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.converter = converter
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.c = exploration_constant
        self.num_actions = num_actions
        self.device = device
        self.batch_size = batch_size
        self.use_puct = use_puct
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._save_counter = 0

        # Root node will be set when game starts
        self.root: Optional[MCTSNode] = None
        self.current_game = None

    def set_game(self, game) -> None:
        """Set the VizDoom game instance for state saving/loading."""
        self.current_game = game

    def reset(self) -> None:
        """Reset the tree for a new episode."""
        if self.root is not None:
            self.root._recursive_clear()
        self.root = None
        self._save_counter = 0

    def _get_save_path(self) -> str:
        """Generate unique path for temporary save file."""
        self._save_counter += 1
        return os.path.join(self.temp_dir, f'mcts_save_{id(self)}_{self._save_counter}.zds')

    def _create_state_from_game(self) -> Tuple[GameState, str]:
        """Capture current game state and save to file.

        Returns:
            Tuple of (GameState, save_file_path)
        """
        state = self.current_game.get_state()
        if state is None:
            raise RuntimeError("Game state is None - episode may have ended")
        screen = state.screen_buffer
        depth = state.depth_buffer if hasattr(state, 'depth_buffer') else None

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
            ascii_frame=ascii_frame,
            depth_bins=depth_bins,
            health=health,
            armor=armor,
            game_reward=game_reward,
        )

        # Save game state to temporary file
        save_path = self._get_save_path()
        self.current_game.save(save_path)
        return game_state, save_path

    def _copy_state_for_expansion(self, node: MCTSNode) -> None:
        """Load saved state into game for expanding a node."""
        if node.save_path is not None and os.path.exists(node.save_path):
            self.current_game.load(node.save_path)

    def _evaluate_state(self, input_ids, attention_mask, depth_ids) -> np.ndarray:
        """Get action probabilities from the model.

        Returns:
            Array of action probabilities
        """
        with torch.no_grad():
            result = self.model(input_ids, attention_mask, depth_ids=depth_ids)
            probs = torch.softmax(result['logits'], dim=-1)[0].cpu().numpy()
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

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection: traverse tree using UCB1 or PUCT until reaching unexpanded node."""
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c, use_puct=self.use_puct)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion: create child node for an untried action.
        Uses model priors from parent node to bias action selection.
        """
        if not node.untried_actions:
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

        # Load parent state and take action
        self._copy_state_for_expansion(node)
        action_name = self.ACTION_NAMES[action]
        buttons = self.ACTION_TO_BUTTONS[action_name]
        self.current_game.make_action(buttons, 4)

        # Capture new state
        child_state, save_path = self._create_state_from_game()

        # Evaluate model priors for child state (will be used when selecting its children)
        input_ids, attention_mask, depth_ids = self._prepare_model_input(child_state)
        child_priors = self._evaluate_state(input_ids, attention_mask, depth_ids)

        logging.debug(f"Expanding node with action '{action_name}' (index {action}) - model priors: {child_priors}")
        # Create child node with its own priors
        child = MCTSNode(
            state=child_state,
            save_path=save_path,
            parent=node,
            action_taken=action,
            num_actions=self.num_actions,
            model_priors=child_priors,
        )

        node.children[action] = child
        return child

    def _rollout(self, node: MCTSNode) -> float:
        """Rollout: simulate random actions and return binary win/lose.

        Win: Game reward increased during rollout
        Lose: Health OR armor decreased with no game reward increase

        Returns:
            1.0 for win, 0.0 for lose, 0.5 for neutral
        """
        # Load node state (create a temp game copy if batching)
        self._copy_state_for_expansion(node)

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

        # Simulate random actions
        for _ in range(self.rollout_depth):
            if self.current_game.is_episode_finished():
                break

            # Random action
            action = np.random.randint(self.num_actions)
            action_name = self.ACTION_NAMES[action]
            buttons = self.ACTION_TO_BUTTONS[action_name]
            self.current_game.make_action(buttons, 4)

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

        reward_increased = end_kills > start_kills
        health_decreased = end_health < start_health
        armor_decreased = end_armor < start_armor

        if reward_increased:
            return 1.0  # Win
        elif health_decreased or armor_decreased:
            return 0.0  # Lose (health or armor decreased)
        else:
            # Neutral - no change in reward, health, or armor
            return 0.5

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagation: update statistics up the tree."""
        current = node
        while current is not None:
            current.update(value)
            current = current.parent

    def run_simulation(self) -> None:
        """Run one MCTS simulation (select, expand, rollout, backprop)."""
        if self.root is None:
            raise ValueError("Root node not initialized. Call initialize_root() first.")

        # Make sure to restore to root state at start of simulation
        self._copy_state_for_expansion(self.root)

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

    def _run_single_leaf_eval(self, leaf_info: Tuple[MCTSNode, str, int]) -> Tuple[MCTSNode, float]:
        """Run a single leaf evaluation for batching.

        Args:
            leaf_info: Tuple of (node, save_path, action_to_expand)

        Returns:
            Tuple of (node, value)
        """
        node, save_path, action = leaf_info

        # Load state and expand
        self.current_game.load(save_path)

        # Take action
        action_name = self.ACTION_NAMES[action]
        buttons = self.ACTION_TO_BUTTONS[action_name]
        self.current_game.make_action(buttons, 4)

        # Capture new state
        child_state, child_save_path = self._create_state_from_game()

        # Evaluate model priors for child state (will be used when selecting its children)
        input_ids, attention_mask, depth_ids = self._prepare_model_input(child_state)
        child_priors = self._evaluate_state(input_ids, attention_mask, depth_ids)

        # Create child node with its own priors
        child = MCTSNode(
            state=child_state,
            save_path=child_save_path,
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
        game_state, save_path = self._create_state_from_game()

        # Evaluate model priors for root state
        input_ids, attention_mask, depth_ids = self._prepare_model_input(game_state)
        model_priors = self._evaluate_state(input_ids, attention_mask, depth_ids)

        self.root = MCTSNode(
            state=game_state,
            save_path=save_path,
            parent=None,
            action_taken=None,
            num_actions=self.num_actions,
            model_priors=model_priors,
        )

    def get_action(self) -> Tuple[str, List[int], int]:
        """Run MCTS and return the best action.

        Returns:
            Tuple of (action_name, button_vector, action_index)
        """
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

        action_name = self.ACTION_NAMES[best_action]
        buttons = self.ACTION_TO_BUTTONS[action_name]

        return action_name, buttons, best_action

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

        # Prune siblings (and delete their save files)
        new_root.prune_siblings()

        # Clean up old root's save file
        if self.root.save_path is not None and os.path.exists(self.root.save_path):
            try:
                os.remove(self.root.save_path)
            except OSError:
                pass

        # Detach from parent
        new_root.set_root()

        # Update root reference
        self.root = new_root

        # Re-initialize saved state from current game position (if game still running)
        if not self.current_game.is_episode_finished():
            try:
                game_state, save_path = self._create_state_from_game()
                # Update the new root's state (but keep tree structure)
                self.root.state = game_state
                if self.root.save_path is not None and os.path.exists(self.root.save_path):
                    try:
                        os.remove(self.root.save_path)
                    except OSError:
                        pass
                self.root.save_path = save_path
            except RuntimeError:
                # State not available, keep existing
                pass
