"""
Project NEXUS - Real-Time GridWorld Renderer
RTX 3060 GPU-Accelerated Visualization for Self-Play Training
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Assume pygame is not available by default to prevent undefined name errors
PYGAME_AVAILABLE = False
try:
    import pygame

    PYGAME_AVAILABLE = True  # Set to True only if import is successful
    print("🎨 Pygame available - full graphical rendering enabled")
except ImportError:
    print("⚠️  Pygame not available - using console visualization mode")


@dataclass
class RenderConfig:
    """Configuration for GridWorld rendering"""
    window_width: int = 1200
    window_height: int = 800
    grid_size: int = 15
    cell_size: int = 40
    fps_target: int = 60

    # Visual settings
    background_color: Tuple[int, int, int] = (20, 20, 30)
    grid_color: Tuple[int, int, int] = (60, 60, 80)

    # Agent colors (team-based)
    team_a_color: Tuple[int, int, int] = (255, 100, 100)  # Red team
    team_b_color: Tuple[int, int, int] = (100, 100, 255)  # Blue team


@dataclass
class VisualizationState:
    """Current state for visualization"""
    agent_positions: List[Tuple[int, int]]
    agent_teams: List[str]
    scores: Dict[str, float]
    step_count: int
    match_info: Dict[str, Any]


class GridWorldRenderer:
    """Main GridWorld real-time renderer with fallback console mode"""

    def __init__(self, config: Optional[RenderConfig] = None):
        # Declare intent to modify the global variable at the top of the method
        global PYGAME_AVAILABLE
        self.config = config or RenderConfig()

        # State management
        self.current_state: Optional[VisualizationState] = None
        self.running = False
        self.render_thread = None
        self.state_queue = queue.Queue(maxsize=10)

        # Performance metrics
        self.frames_rendered = 0
        self.start_time = time.time()

        # Initialize pygame if available
        if PYGAME_AVAILABLE:
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((
                    self.config.window_width,
                    self.config.window_height
                ))
                pygame.display.set_caption("Project NEXUS - Self-Play Training")
                self.clock = pygame.time.Clock()
                print("🎨 GridWorld Renderer initialized with Pygame graphics")
            except Exception as e:
                print(f"⚠️  Pygame initialization failed: {e}")
                PYGAME_AVAILABLE = False
                print("🔄 Falling back to console mode")

        if not PYGAME_AVAILABLE:
            print("🎨 GridWorld Renderer initialized (console mode)")
            print("💡 Install pygame with: conda install -c conda-forge pygame")

    def start_rendering(self):
        """Start the rendering loop in separate thread"""
        if self.running:
            return

        self.running = True
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()

        if PYGAME_AVAILABLE:
            print("✅ Graphical rendering started - window should appear")
        else:
            print("✅ Console rendering started - watch terminal for updates")

    def stop_rendering(self):
        """Stop the rendering loop"""
        self.running = False
        if self.render_thread:
            self.render_thread.join(timeout=2.0)

        if PYGAME_AVAILABLE:
            try:
                pygame.quit()
            except:
                pass
        print("🛑 Rendering stopped")

    def update_state(self, env_state: Dict[str, Any]):
        """Update visualization state from environment"""
        try:
            # Extract agent information
            agent_positions = []
            agent_teams = []

            if 'env_state' in env_state and 'agents' in env_state['env_state']:
                for agent_info in env_state['env_state']['agents']:
                    pos = agent_info.get('position', (0, 0))
                    team = agent_info.get('team', 'neutral')
                    agent_positions.append(pos)
                    agent_teams.append(team)
            else:
                # Default positions for demo
                step = env_state.get('step', 0)
                agent_positions = [
                    (1 + step % 5, 1 + step % 5),
                    (2 + step % 5, 2 + step % 5),
                    (13 - step % 5, 13 - step % 5),
                    (12 - step % 5, 12 - step % 5)
                ]
                agent_teams = ['team_a', 'team_a', 'team_b', 'team_b']

            # Get scores
            scores = {
                'team_a': env_state.get('team_a_score', 0),
                'team_b': env_state.get('team_b_score', 0)
            }

            step_count = env_state.get('step', 0)
            match_info = env_state.get('match_info', {})

            # Create visualization state
            vis_state = VisualizationState(
                agent_positions=agent_positions,
                agent_teams=agent_teams,
                scores=scores,
                step_count=step_count,
                match_info=match_info
            )

            # Queue state update (non-blocking)
            try:
                self.state_queue.put_nowait(vis_state)
            except queue.Full:
                pass  # Skip frame if queue is full

        except Exception as e:
            print(f"⚠️  Error updating visualization state: {e}")

    def _render_loop(self):
        """Main rendering loop (runs in separate thread)"""

        while self.running:
            try:
                # Handle pygame events if available
                if PYGAME_AVAILABLE:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            self.running = False
                            break

                # Get latest state
                try:
                    while not self.state_queue.empty():
                        self.current_state = self.state_queue.get_nowait()
                except queue.Empty:
                    pass

                # Render frame
                if PYGAME_AVAILABLE:
                    self._render_pygame()
                else:
                    self._render_console()

                self.frames_rendered += 1
                time.sleep(1.0 / self.config.fps_target)

            except Exception as e:
                print(f"⚠️  Rendering error: {e}")
                time.sleep(0.1)  # Prevent tight error loop

        print("🎬 Render loop ended")

    def _render_pygame(self):
        """Render using pygame"""
        if not PYGAME_AVAILABLE:
            return

        try:
            # Clear screen
            self.screen.fill(self.config.background_color)

            if self.current_state:
                # Draw grid
                self._draw_grid()

                # Draw agents
                self._draw_agents()

                # Draw UI
                self._draw_ui()
            else:
                self._draw_waiting_screen()

            # Update display
            pygame.display.flip()
            self.clock.tick(self.config.fps_target)
        except Exception as e:
            print(f"⚠️  Pygame rendering error: {e}")

    def _render_console(self):
        """Render using console output (fallback)"""
        if self.current_state and self.frames_rendered % 30 == 0:  # Update every 0.5 seconds
            print(f"🎮 Step {self.current_state.step_count:4d}: "
                  f"🔴 Team A: {self.current_state.scores.get('team_a', 0):6.1f} | "
                  f"🔵 Team B: {self.current_state.scores.get('team_b', 0):6.1f} | "
                  f"🎨 FPS: {self.get_current_fps():.1f}")

            # Show agent positions occasionally
            if self.frames_rendered % 180 == 0:  # Every 3 seconds
                print("📍 Agent positions:")
                for i, (pos, team) in enumerate(zip(self.current_state.agent_positions,
                                                    self.current_state.agent_teams)):
                    color_emoji = "🔴" if team == "team_a" else "🔵"
                    print(f"   {color_emoji} Agent {i}: {pos}")

    def _draw_grid(self):
        """Draw grid background"""
        if not PYGAME_AVAILABLE:
            return

        try:
            cell_size = self.config.cell_size

            # Calculate grid offset
            grid_pixel_size = self.config.grid_size * cell_size
            offset_x = (self.config.window_width - grid_pixel_size) // 2
            offset_y = (self.config.window_height - grid_pixel_size) // 2

            # Vertical lines
            for x in range(self.config.grid_size + 1):
                start_pos = (offset_x + x * cell_size, offset_y)
                end_pos = (offset_x + x * cell_size, offset_y + grid_pixel_size)
                pygame.draw.line(self.screen, self.config.grid_color, start_pos, end_pos, 1)

            # Horizontal lines
            for y in range(self.config.grid_size + 1):
                start_pos = (offset_x, offset_y + y * cell_size)
                end_pos = (offset_x + grid_pixel_size, offset_y + y * cell_size)
                pygame.draw.line(self.screen, self.config.grid_color, start_pos, end_pos, 1)
        except Exception as e:
            print(f"⚠️  Grid drawing error: {e}")

    def _draw_agents(self):
        """Draw agents with team colors"""
        if not PYGAME_AVAILABLE or not self.current_state:
            return

        try:
            cell_size = self.config.cell_size
            grid_pixel_size = self.config.grid_size * cell_size
            offset_x = (self.config.window_width - grid_pixel_size) // 2
            offset_y = (self.config.window_height - grid_pixel_size) // 2

            team_colors = {
                'team_a': self.config.team_a_color,
                'team_b': self.config.team_b_color,
                'neutral': (255, 255, 255)
            }

            for i, (pos, team) in enumerate(zip(self.current_state.agent_positions,
                                                self.current_state.agent_teams)):
                x, y = pos

                # Clamp positions to grid bounds
                x = max(0, min(x, self.config.grid_size - 1))
                y = max(0, min(y, self.config.grid_size - 1))

                pixel_x = offset_x + x * cell_size + cell_size // 2
                pixel_y = offset_y + y * cell_size + cell_size // 2

                color = team_colors.get(team, (255, 255, 255))

                # Agent body (circle)
                pygame.draw.circle(self.screen, color, (pixel_x, pixel_y), 8)
                pygame.draw.circle(self.screen, (255, 255, 255), (pixel_x, pixel_y), 8, 2)

                # Agent ID
                font = pygame.font.Font(None, 16)
                text = font.render(str(i), True, (255, 255, 255))
                text_rect = text.get_rect(center=(pixel_x, pixel_y))
                self.screen.blit(text, text_rect)
        except Exception as e:
            print(f"⚠️  Agent drawing error: {e}")

    def _draw_ui(self):
        """Draw UI elements (scores, info, FPS)"""
        if not PYGAME_AVAILABLE or not self.current_state:
            return

        try:
            font = pygame.font.Font(None, 24)
            small_font = pygame.font.Font(None, 18)

            # Scores
            y_offset = 10
            colors = {'team_a': self.config.team_a_color, 'team_b': self.config.team_b_color}

            for team, score in self.current_state.scores.items():
                color = colors.get(team, (255, 255, 255))
                score_text = f"{team.upper()}: {score:.1f}"
                text_surface = font.render(score_text, True, color)
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 30

            # Step count
            step_text = f"Step: {self.current_state.step_count}"
            text_surface = small_font.render(step_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, self.config.window_height - 60))

            # FPS
            fps = self.get_current_fps()
            fps_text = f"FPS: {fps:.1f}"
            text_surface = small_font.render(fps_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, self.config.window_height - 40))

            # Instructions
            instruction_text = "Press ESC to close window"
            text_surface = small_font.render(instruction_text, True, (150, 150, 150))
            self.screen.blit(text_surface, (10, self.config.window_height - 20))

        except Exception as e:
            print(f"⚠️  UI drawing error: {e}")

    def _draw_waiting_screen(self):
        """Render waiting screen when no state available"""
        if not PYGAME_AVAILABLE:
            return

        try:
            font = pygame.font.Font(None, 48)
            text = font.render("NEXUS Self-Play Training", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.config.window_width // 2, self.config.window_height // 2))
            self.screen.blit(text, text_rect)

            small_font = pygame.font.Font(None, 24)
            waiting_text = small_font.render("Waiting for training data...", True, (200, 200, 200))
            waiting_rect = waiting_text.get_rect(
                center=(self.config.window_width // 2, self.config.window_height // 2 + 50))
            self.screen.blit(waiting_text, waiting_rect)
        except Exception as e:
            print(f"⚠️  Waiting screen drawing error: {e}")

    def get_current_fps(self) -> float:
        """Calculate current FPS"""
        runtime = time.time() - self.start_time
        if runtime > 0:
            return min(self.frames_rendered / runtime, self.config.fps_target)
        return 0.0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics"""
        runtime = time.time() - self.start_time
        avg_fps = self.frames_rendered / max(runtime, 1)

        return {
            'frames_rendered': self.frames_rendered,
            'runtime_seconds': runtime,
            'average_fps': avg_fps,
            'current_fps': self.get_current_fps(),
            'pygame_available': PYGAME_AVAILABLE
        }


def create_single_match_renderer(rtx_optimized: bool = True) -> GridWorldRenderer:
    """Create single match renderer with RTX optimization"""
    config = RenderConfig()
    if rtx_optimized:
        config.fps_target = 60

    return GridWorldRenderer(config)


# Demo function for testing
async def demo_renderer():
    """Demo function that works with or without pygame"""
    print("🎬 Starting renderer demo...")

    renderer = create_single_match_renderer()
    renderer.start_rendering()

    # Simulate match states
    try:
        for step in range(1000):
            # Check if rendering should continue
            if not renderer.running:
                break

            dummy_state = {
                'team_a_score': step * 0.15,
                'team_b_score': step * 0.12,
                'step': step,
                'match_info': {'match_id': 'demo_match'}
            }

            renderer.update_state(dummy_state)
            await asyncio.sleep(0.016)  # ~60 FPS update rate

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    finally:
        renderer.stop_rendering()
        print("🎬 Demo complete!")


if __name__ == "__main__":
    import asyncio

    # The main entry point now uses a unified demo function
    # that handles both graphical and console modes gracefully.
    try:
        asyncio.run(demo_renderer())
    except KeyboardInterrupt:
        print("\nExiting.")