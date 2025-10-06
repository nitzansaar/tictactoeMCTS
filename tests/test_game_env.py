"""
Unit tests for GameEnv class.

Tests cover:
- Horizontal, vertical, diagonal win detection
- Draw conditions
- Edge cases (different board sizes, K values)
- Legal move validation
- Board encoding
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.game_env import GameEnv


class TestGameEnvBasic(unittest.TestCase):
    """Test basic game functionality."""
    
    def test_initialization(self):
        """Test game initialization with different parameters."""
        env = GameEnv(3, 3)
        self.assertEqual(env.n, 3)
        self.assertEqual(env.k, 3)
        self.assertEqual(env.current_player, 1)
        self.assertFalse(env.game_over)
        self.assertIsNone(env.winner)
        
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with self.assertRaises(ValueError):
            GameEnv(3, 4)  # K > N
        with self.assertRaises(ValueError):
            GameEnv(0, 0)  # N < 1
        with self.assertRaises(ValueError):
            GameEnv(3, 0)  # K < 1
    
    def test_reset(self):
        """Test game reset functionality."""
        env = GameEnv(3, 3)
        env.apply_move(0, 0)
        env.apply_move(1, 1)
        
        board = env.reset()
        
        np.testing.assert_array_equal(board, np.zeros((3, 3)))
        self.assertEqual(env.current_player, 1)
        self.assertEqual(env.move_count, 0)
        self.assertFalse(env.game_over)
    
    def test_legal_moves(self):
        """Test legal move generation."""
        env = GameEnv(3, 3)
        
        # Initially all moves are legal
        legal_moves = env.get_legal_moves()
        self.assertEqual(len(legal_moves), 9)
        
        # After one move, 8 remain
        env.apply_move(0, 0)
        legal_moves = env.get_legal_moves()
        self.assertEqual(len(legal_moves), 8)
        self.assertNotIn((0, 0), legal_moves)
        
    def test_illegal_moves(self):
        """Test that illegal moves raise errors."""
        env = GameEnv(3, 3)
        
        # Out of bounds
        with self.assertRaises(ValueError):
            env.apply_move(3, 3)
        with self.assertRaises(ValueError):
            env.apply_move(-1, 0)
        
        # Already occupied
        env.apply_move(0, 0)
        with self.assertRaises(ValueError):
            env.apply_move(0, 0)
    
    def test_player_switching(self):
        """Test that players alternate correctly."""
        env = GameEnv(3, 3)
        
        self.assertEqual(env.current_player, 1)
        env.apply_move(0, 0)
        self.assertEqual(env.current_player, -1)
        env.apply_move(1, 1)
        self.assertEqual(env.current_player, 1)


class TestWinDetection(unittest.TestCase):
    """Test win detection for various patterns."""
    
    def test_horizontal_win(self):
        """Test horizontal win detection."""
        env = GameEnv(3, 3)
        
        # Player 1 wins horizontally on row 0
        env.apply_move(0, 0)  # X
        env.apply_move(1, 0)  # O
        env.apply_move(0, 1)  # X
        env.apply_move(1, 1)  # O
        _, reward, done = env.apply_move(0, 2)  # X wins
        
        self.assertTrue(done)
        self.assertEqual(env.winner, 1)
        self.assertEqual(reward, 1.0)
    
    def test_vertical_win(self):
        """Test vertical win detection."""
        env = GameEnv(3, 3)
        
        # Player -1 wins vertically on column 1
        env.apply_move(0, 0)  # X
        env.apply_move(0, 1)  # O
        env.apply_move(1, 0)  # X
        env.apply_move(1, 1)  # O
        env.apply_move(2, 2)  # X
        _, reward, done = env.apply_move(2, 1)  # O wins
        
        self.assertTrue(done)
        self.assertEqual(env.winner, -1)
        self.assertEqual(reward, 1.0)  # Reward is from current player's perspective
    
    def test_diagonal_win(self):
        """Test main diagonal win detection."""
        env = GameEnv(3, 3)
        
        # Player 1 wins on main diagonal
        env.apply_move(0, 0)  # X
        env.apply_move(0, 1)  # O
        env.apply_move(1, 1)  # X
        env.apply_move(0, 2)  # O
        _, reward, done = env.apply_move(2, 2)  # X wins
        
        self.assertTrue(done)
        self.assertEqual(env.winner, 1)
        self.assertEqual(reward, 1.0)
    
    def test_anti_diagonal_win(self):
        """Test anti-diagonal win detection."""
        env = GameEnv(3, 3)
        
        # Player 1 wins on anti-diagonal
        env.apply_move(0, 2)  # X
        env.apply_move(0, 0)  # O
        env.apply_move(1, 1)  # X
        env.apply_move(0, 1)  # O
        _, reward, done = env.apply_move(2, 0)  # X wins
        
        self.assertTrue(done)
        self.assertEqual(env.winner, 1)
    
    def test_no_premature_win(self):
        """Test that incomplete lines don't trigger a win."""
        env = GameEnv(3, 3)
        
        env.apply_move(0, 0)  # X
        env.apply_move(1, 1)  # O
        
        self.assertFalse(env.game_over)
        self.assertIsNone(env.winner)


class TestDrawCondition(unittest.TestCase):
    """Test draw detection."""
    
    def test_draw_game(self):
        """Test that a full board with no winner is a draw."""
        env = GameEnv(3, 3)
        
        # Create a draw scenario
        # Final board:
        # X X O
        # O O X
        # X O X
        moves = [
            (0, 0),  # Move 1: X
            (0, 2),  # Move 2: O
            (1, 2),  # Move 3: X
            (1, 0),  # Move 4: O
            (2, 0),  # Move 5: X
            (1, 1),  # Move 6: O
            (2, 2),  # Move 7: X
            (2, 1),  # Move 8: O
            (0, 1),  # Move 9: X (last move, fills board)
        ]
        
        for i, (row, col) in enumerate(moves[:-1]):
            _, _, done = env.apply_move(row, col)
            self.assertFalse(done, f"Game ended prematurely at move {i+1}")
        
        # Last move should result in draw
        _, reward, done = env.apply_move(*moves[-1])
        
        self.assertTrue(done)
        self.assertIsNone(env.winner)
        self.assertEqual(reward, 0.0)
        self.assertTrue(env.is_draw())


class TestLargerBoards(unittest.TestCase):
    """Test functionality on larger boards and different K values."""
    
    def test_5x5_board(self):
        """Test 5×5 board with K=4."""
        env = GameEnv(5, 4)
        
        # Test horizontal win with K=4
        env.apply_move(0, 0)  # X
        env.apply_move(1, 0)  # O
        env.apply_move(0, 1)  # X
        env.apply_move(1, 1)  # O
        env.apply_move(0, 2)  # X
        env.apply_move(1, 2)  # O
        _, reward, done = env.apply_move(0, 3)  # X wins with 4 in a row
        
        self.assertTrue(done)
        self.assertEqual(env.winner, 1)
    
    def test_k_less_than_n(self):
        """Test that K < N works correctly."""
        env = GameEnv(5, 3)
        
        # Win with 3 in a row on a 5×5 board
        env.apply_move(0, 0)  # X
        env.apply_move(1, 0)  # O
        env.apply_move(0, 1)  # X
        env.apply_move(1, 1)  # O
        _, reward, done = env.apply_move(0, 2)  # X wins with 3 in a row
        
        self.assertTrue(done)
        self.assertEqual(env.winner, 1)
    
    def test_diagonal_on_larger_board(self):
        """Test diagonal win on larger board."""
        env = GameEnv(5, 4)
        
        # Create diagonal win
        env.apply_move(0, 0)  # X
        env.apply_move(0, 1)  # O
        env.apply_move(1, 1)  # X
        env.apply_move(0, 2)  # O
        env.apply_move(2, 2)  # X
        env.apply_move(0, 3)  # O
        _, reward, done = env.apply_move(3, 3)  # X wins diagonally
        
        self.assertTrue(done)
        self.assertEqual(env.winner, 1)


class TestBoardEncoding(unittest.TestCase):
    """Test board encoding for neural network input."""
    
    def test_encoding_shape(self):
        """Test that encoding has correct shape."""
        env = GameEnv(3, 3)
        encoding = env.get_board_encoding()
        
        self.assertEqual(encoding.shape, (3, 3, 3))
        self.assertEqual(encoding.dtype, np.float32)
    
    def test_encoding_content(self):
        """Test that encoding correctly represents board state."""
        env = GameEnv(3, 3)
        env.apply_move(0, 0)  # X at (0, 0)
        env.apply_move(1, 1)  # O at (1, 1)
        
        encoding = env.get_board_encoding()
        
        # Plane 0: Player 1 (X)
        self.assertEqual(encoding[0, 0, 0], 1.0)
        self.assertEqual(encoding[0, 1, 1], 0.0)
        
        # Plane 1: Player 2 (O)
        self.assertEqual(encoding[1, 0, 0], 0.0)
        self.assertEqual(encoding[1, 1, 1], 1.0)
        
        # Plane 2: Empty
        self.assertEqual(encoding[2, 0, 0], 0.0)
        self.assertEqual(encoding[2, 1, 1], 0.0)
        self.assertEqual(encoding[2, 2, 2], 1.0)
    
    def test_canonical_board(self):
        """Test canonical board representation."""
        env = GameEnv(3, 3)
        env.apply_move(0, 0)  # X (player 1)
        
        # From player -1's perspective, X should be -1
        env.current_player = -1
        canonical = env.get_canonical_board()
        
        self.assertEqual(canonical[0, 0], -1)


class TestGameCloning(unittest.TestCase):
    """Test game state cloning."""
    
    def test_clone(self):
        """Test that clone creates independent copy."""
        env = GameEnv(3, 3)
        env.apply_move(0, 0)
        env.apply_move(1, 1)
        
        clone = env.clone()
        
        # Clone should have same state
        np.testing.assert_array_equal(clone.board, env.board)
        self.assertEqual(clone.current_player, env.current_player)
        self.assertEqual(clone.move_count, env.move_count)
        
        # Modifying clone shouldn't affect original
        clone.apply_move(2, 2)
        self.assertNotEqual(clone.move_count, env.move_count)


class TestRendering(unittest.TestCase):
    """Test board rendering."""
    
    def test_render(self):
        """Test that board renders correctly."""
        env = GameEnv(3, 3)
        env.apply_move(0, 0)  # X
        env.apply_move(1, 1)  # O
        
        rendered = env.render()
        
        self.assertIn('X', rendered)
        self.assertIn('O', rendered)
        self.assertIn('.', rendered)
    
    def test_str_and_repr(self):
        """Test __str__ and __repr__ methods."""
        env = GameEnv(3, 3)
        
        str_repr = str(env)
        self.assertIsInstance(str_repr, str)
        
        repr_str = repr(env)
        self.assertIn('GameEnv', repr_str)
        self.assertIn('n=3', repr_str)


if __name__ == '__main__':
    unittest.main()

