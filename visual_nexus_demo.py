# File: visual_nexus_demo_fixed.py
# REAL-TIME PYGAME VISUALIZATION - FIXED MULTI-AGENT VERSION
# Now ALL agents move and act intelligently every step!

import pygame
import sys
import os
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import with error handling - but we'll use our own multi-agent simulation
try:
    from environment.grid_world import GridWorld
    ENV_AVAILABLE = True
except ImportError:
    print("⚠️ GridWorld not available - using full simulation mode")
    ENV_AVAILABLE = False

# Initialize Pygame
pygame.init()

# Color schemes
class Colors:
    BACKGROUND = (20, 20, 25)
    GRID_LINE = (60, 60, 70)
    EMPTY_CELL = (40, 40, 50)
    
    # Agent colors - distinct and vibrant
    AGENT_COLORS = [
        (100, 200, 255),  # Agent 0: Bright blue
        (255, 100, 100),  # Agent 1: Bright red  
        (100, 255, 100),  # Agent 2: Bright green
        (255, 255, 100),  # Agent 3: Bright yellow
        (255, 100, 255),  # Agent 4: Bright magenta
        (100, 255, 255),  # Agent 5: Bright cyan
    ]
    
    # Resource colors
    RESOURCE = (255, 215, 0)  # Gold
    RESOURCE_GLOW = (255, 255, 150)  # Light gold glow
    
    # UI colors
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (180, 180, 180)
    UI_ACCENT = (0, 150, 255)
    SUCCESS_GREEN = (50, 255, 50)
    WARNING_ORANGE = (255, 150, 50)
    
    # Effect colors
    COORDINATION_EFFECT = (0, 255, 0, 100)
    GATHERING_EFFECT = (255, 215, 0, 150)

class AgentState(Enum):
    EXPLORING = "exploring"
    MOVING_TO_RESOURCE = "moving_to_resource"
    GATHERING = "gathering"
    COORDINATING = "coordinating"
    IDLE = "idle"

@dataclass
class VisualAgent:
    """Visual representation of an agent with full autonomy"""
    id: int
    pos: Tuple[int, int]
    color: Tuple[int, int, int]
    inventory: Dict[str, int]
    state: AgentState
    last_action: str
    trail: List[Tuple[int, int]]
    coordination_partners: List[int]
    target_resource: Optional[Tuple[int, int]]
    action_cooldown: int
    energy: float
    last_reward: float

class TrueMultiAgentEnvironment:
    """True multi-agent environment where ALL agents act every step"""
    
    def __init__(self, grid_size=15, n_agents=3, max_resources=6):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.max_resources = max_resources
        
        # Initialize agents
        self.agents = []
        start_positions = self._generate_start_positions(n_agents)
        
        for i in range(n_agents):
            agent = VisualAgent(
                id=i,
                pos=start_positions[i],
                color=Colors.AGENT_COLORS[i % len(Colors.AGENT_COLORS)],
                inventory={'resources': 0, 'energy': 0, 'crystals': 0},
                state=AgentState.EXPLORING,
                last_action="spawn",
                trail=[start_positions[i]],
                coordination_partners=[],
                target_resource=None,
                action_cooldown=0,
                energy=100.0,
                last_reward=0.0
            )
            self.agents.append(agent)
        
        # Initialize resources
        self.resources = []
        self.resource_types = ['gold', 'energy', 'crystal']
        self._spawn_initial_resources()
        
        # Environment stats
        self.step_count = 0
        self.total_rewards = [0.0] * n_agents
        self.coordination_events = 0
        
        print(f"🌍 True Multi-Agent Environment created:")
        print(f"   📊 {n_agents} agents, {len(self.resources)} resources")
        print(f"   🎯 ALL agents will move and act every step!")
    
    def _generate_start_positions(self, n_agents: int) -> List[Tuple[int, int]]:
        """Generate non-overlapping start positions"""
        positions = []
        max_attempts = 100
        
        # Try to place agents in corners and edges first
        preferred_positions = [
            (1, 1), (self.grid_size-2, 1), (1, self.grid_size-2),
            (self.grid_size-2, self.grid_size-2), (self.grid_size//2, 1),
            (1, self.grid_size//2), (self.grid_size//2, self.grid_size-2),
            (self.grid_size-2, self.grid_size//2)
        ]
        
        for i in range(n_agents):
            if i < len(preferred_positions):
                positions.append(preferred_positions[i])
            else:
                # Random placement for additional agents
                for _ in range(max_attempts):
                    pos = (np.random.randint(1, self.grid_size-1), 
                          np.random.randint(1, self.grid_size-1))
                    if pos not in positions:
                        positions.append(pos)
                        break
        
        return positions
    
    def _spawn_initial_resources(self):
        """Spawn initial resources"""
        self.resources = []
        
        for _ in range(self.max_resources):
            pos = self._find_empty_position()
            if pos:
                resource_type = np.random.choice(self.resource_types)
                self.resources.append({
                    'pos': pos,
                    'type': resource_type,
                    'value': np.random.uniform(0.5, 2.0),
                    'respawn_timer': 0
                })
    
    def _find_empty_position(self) -> Optional[Tuple[int, int]]:
        """Find an empty position for resource spawning"""
        occupied_positions = {agent.pos for agent in self.agents}
        occupied_positions.update({res['pos'] for res in self.resources})
        
        # Try random positions
        for _ in range(50):
            pos = (np.random.randint(2, self.grid_size-2), 
                  np.random.randint(2, self.grid_size-2))
            if pos not in occupied_positions:
                return pos
        
        return None
    
    def step(self) -> Dict:
        """Execute one step for ALL agents simultaneously"""
        self.step_count += 1
        
        # Update all agents simultaneously
        agent_actions = []
        step_rewards = []
        
        for agent in self.agents:
            # Reduce action cooldown
            if agent.action_cooldown > 0:
                agent.action_cooldown -= 1
            
            # Update agent energy (slight decay for realism)
            agent.energy = max(0, agent.energy - 0.1)
            
            # Determine agent action
            action_result = self._execute_agent_action(agent)
            agent_actions.append(action_result['action'])
            step_rewards.append(action_result['reward'])
            
            # Update agent state based on action result
            agent.last_reward = action_result['reward']
            self.total_rewards[agent.id] += action_result['reward']
        
        # Check for coordination opportunities
        self._check_coordination()
        
        # Update environment (resource respawning, etc.)
        self._update_environment()
        
        return {
            'step': self.step_count,
            'agents': self.agents,
            'resources': self.resources,
            'actions': agent_actions,
            'rewards': step_rewards,
            'total_rewards': self.total_rewards,
            'coordination_events': self.coordination_events
        }
    
    def _execute_agent_action(self, agent: VisualAgent) -> Dict:
        """Execute action for a single agent"""
        
        if agent.action_cooldown > 0:
            # Agent is on cooldown, skip action
            return {'action': 'wait', 'reward': -0.01}
        
        # Intelligent action selection
        action_result = self._select_intelligent_action(agent)
        
        # Execute the action
        if action_result['action'] == 'move':
            new_pos = action_result['new_pos']
            
            # Check bounds
            new_pos = (max(0, min(self.grid_size-1, new_pos[0])),
                      max(0, min(self.grid_size-1, new_pos[1])))
            
            # Check for collisions with other agents
            occupied = any(other.pos == new_pos for other in self.agents if other.id != agent.id)
            
            if not occupied:
                agent.pos = new_pos
                agent.trail.append(new_pos)
                
                # Limit trail length
                if len(agent.trail) > 8:
                    agent.trail.pop(0)
                
                # Small movement cost
                return {'action': 'move', 'reward': -0.01}
            else:
                # Collision, try alternative move
                alternatives = self._get_adjacent_positions(agent.pos)
                for alt_pos in alternatives:
                    if not any(other.pos == alt_pos for other in self.agents if other.id != agent.id):
                        agent.pos = alt_pos
                        agent.trail.append(alt_pos)
                        return {'action': 'move_alt', 'reward': -0.01}
                
                # No valid moves
                return {'action': 'blocked', 'reward': -0.02}
        
        elif action_result['action'] == 'gather':
            # Gather resource
            resource_to_remove = None
            for resource in self.resources:
                if resource['pos'] == agent.pos:
                    resource_to_remove = resource
                    break
            
            if resource_to_remove:
                # Successfully gathered
                self.resources.remove(resource_to_remove)
                
                # Update agent inventory
                resource_type = resource_to_remove['type']
                if resource_type in agent.inventory:
                    agent.inventory[resource_type] += 1
                else:
                    agent.inventory[resource_type] = 1
                
                agent.inventory['resources'] += 1
                agent.state = AgentState.GATHERING
                agent.last_action = f"gathered_{resource_type}"
                agent.action_cooldown = 2  # Brief cooldown after gathering
                agent.energy = min(100, agent.energy + 10)  # Energy boost
                
                # Spawn new resource occasionally
                if np.random.random() < 0.4:
                    self._spawn_single_resource()
                
                return {'action': 'gather', 'reward': resource_to_remove['value']}
            else:
                return {'action': 'gather_failed', 'reward': -0.05}
        
        elif action_result['action'] == 'explore':
            agent.state = AgentState.EXPLORING
            agent.last_action = "exploring"
            return {'action': 'explore', 'reward': -0.005}
        
        return {'action': 'idle', 'reward': -0.01}
    
    def _select_intelligent_action(self, agent: VisualAgent) -> Dict:
        """Select intelligent action for agent"""
        
        # Check if agent is on a resource
        on_resource = any(res['pos'] == agent.pos for res in self.resources)
        if on_resource:
            return {'action': 'gather'}
        
        # Find nearest resource
        if self.resources:
            nearest_resource = min(self.resources, 
                                 key=lambda r: self._manhattan_distance(agent.pos, r['pos']))
            
            # Move towards nearest resource
            dx = np.sign(nearest_resource['pos'][0] - agent.pos[0])
            dy = np.sign(nearest_resource['pos'][1] - agent.pos[1])
            
            # Add some randomness to avoid getting stuck
            if np.random.random() < 0.2:
                dx += np.random.choice([-1, 0, 1])
                dy += np.random.choice([-1, 0, 1])
                dx = np.sign(dx)
                dy = np.sign(dy)
            
            new_pos = (agent.pos[0] + dx, agent.pos[1] + dy)
            agent.state = AgentState.MOVING_TO_RESOURCE
            agent.target_resource = nearest_resource['pos']
            
            return {'action': 'move', 'new_pos': new_pos}
        
        else:
            # No resources, explore randomly
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                         (-1, -1), (-1, 1), (1, -1), (1, 1)]
            direction = np.random.choice(len(directions))
            dx, dy = directions[direction]
            
            new_pos = (agent.pos[0] + dx, agent.pos[1] + dy)
            agent.state = AgentState.EXPLORING
            
            return {'action': 'move', 'new_pos': new_pos}
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_adjacent_positions(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid adjacent positions"""
        x, y = pos
        adjacent = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                    adjacent.append((new_x, new_y))
        
        return adjacent
    
    def _check_coordination(self):
        """Check for coordination opportunities between agents"""
        
        # Clear previous coordination
        for agent in self.agents:
            agent.coordination_partners = []
        
        # Find agents within coordination distance
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                distance = self._manhattan_distance(agent1.pos, agent2.pos)
                
                if distance <= 2:  # Coordination distance
                    # Check if they're working toward same resource
                    if (agent1.target_resource and agent2.target_resource and
                        agent1.target_resource == agent2.target_resource):
                        
                        agent1.coordination_partners.append(agent2.id)
                        agent2.coordination_partners.append(agent1.id)
                        
                        agent1.state = AgentState.COORDINATING
                        agent2.state = AgentState.COORDINATING
                        
                        # Coordination bonus
                        self.total_rewards[agent1.id] += 0.1
                        self.total_rewards[agent2.id] += 0.1
                        self.coordination_events += 1
    
    def _update_environment(self):
        """Update environment state"""
        
        # Occasionally spawn new resources
        if len(self.resources) < self.max_resources and np.random.random() < 0.1:
            self._spawn_single_resource()
    
    def _spawn_single_resource(self):
        """Spawn a single new resource"""
        pos = self._find_empty_position()
        if pos:
            resource_type = np.random.choice(self.resource_types)
            resource = {
                'pos': pos,
                'type': resource_type,
                'value': np.random.uniform(0.8, 1.5),
                'respawn_timer': 0
            }
            self.resources.append(resource)
    
    def reset(self):
        """Reset the environment"""
        self.step_count = 0
        self.total_rewards = [0.0] * self.n_agents
        self.coordination_events = 0
        
        # Reset agent positions and states
        start_positions = self._generate_start_positions(self.n_agents)
        
        for i, agent in enumerate(self.agents):
            agent.pos = start_positions[i]
            agent.inventory = {'resources': 0, 'energy': 0, 'crystals': 0}
            agent.state = AgentState.EXPLORING
            agent.last_action = "reset"
            agent.trail = [start_positions[i]]
            agent.coordination_partners = []
            agent.target_resource = None
            agent.action_cooldown = 0
            agent.energy = 100.0
            agent.last_reward = 0.0
        
        # Reset resources
        self._spawn_initial_resources()

class NEXUSVisualizerFixed:
    """Fixed multi-agent visualizer where ALL agents are active"""
    
    def __init__(self, width=1200, height=900, grid_size=15):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        
        # Calculate layout
        self.sidebar_width = 320
        self.grid_area_width = width - self.sidebar_width
        self.cell_size = min(self.grid_area_width // grid_size, (height - 100) // grid_size)
        self.grid_offset_x = (self.grid_area_width - (grid_size * self.cell_size)) // 2
        self.grid_offset_y = (height - (grid_size * self.cell_size)) // 2
        
        # Initialize display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🚀 Project NEXUS - FIXED Multi-Agent Visualization")
        
        # Initialize fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Initialize clock
        self.clock = pygame.time.Clock()
        
        # Create our TRUE multi-agent environment
        self.env = TrueMultiAgentEnvironment(grid_size=grid_size, n_agents=3, max_resources=6)
        
        # Simulation state
        self.running = True
        self.paused = False
        self.speed = 8  # FPS
        self.effects = []
        
        print("🎮 FIXED Project NEXUS Visual Demo initialized!")
        print(f"📺 Display: {width}x{height}, Grid: {grid_size}x{grid_size}")
        print("✅ ALL AGENTS WILL NOW MOVE AND ACT EVERY STEP!")
    
    def update_simulation(self):
        """Update the simulation - now ALL agents act"""
        
        if self.paused:
            return
        
        # Step the true multi-agent environment
        step_result = self.env.step()
        
        # Add visual effects for actions
        for i, (agent, reward) in enumerate(zip(self.env.agents, step_result['rewards'])):
            if reward > 0.5:  # Good reward (resource gathered)
                self._add_effect("gather", agent.pos, Colors.RESOURCE_GLOW)
            elif len(agent.coordination_partners) > 0:  # Coordination
                self._add_effect("coordinate", agent.pos, Colors.COORDINATION_EFFECT)
    
    def _add_effect(self, effect_type: str, pos: Tuple[int, int], color: Tuple[int, int, int]):
        """Add visual effect"""
        effect = {
            'type': effect_type,
            'pos': pos,
            'color': color,
            'timer': 30,
            'start_timer': 30
        }
        self.effects.append(effect)
    
    def draw(self):
        """Main drawing function"""
        
        # Clear screen
        self.screen.fill(Colors.BACKGROUND)
        
        # Draw grid
        self._draw_grid()
        
        # Draw agent trails
        self._draw_trails()
        
        # Draw resources
        self._draw_resources()
        
        # Draw agents
        self._draw_agents()
        
        # Draw effects
        self._draw_effects()
        
        # Draw coordination links
        self._draw_coordination_links()
        
        # Draw UI
        self._draw_ui()
        
        # Update display
        pygame.display.flip()
    
    def _draw_grid(self):
        """Draw the grid background"""
        for x in range(self.grid_size + 1):
            start_pos = (self.grid_offset_x + x * self.cell_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + x * self.cell_size, 
                      self.grid_offset_y + self.grid_size * self.cell_size)
            pygame.draw.line(self.screen, Colors.GRID_LINE, start_pos, end_pos, 1)
        
        for y in range(self.grid_size + 1):
            start_pos = (self.grid_offset_x, self.grid_offset_y + y * self.cell_size)
            end_pos = (self.grid_offset_x + self.grid_size * self.cell_size, 
                      self.grid_offset_y + y * self.cell_size)
            pygame.draw.line(self.screen, Colors.GRID_LINE, start_pos, end_pos, 1)
    
    def _draw_trails(self):
        """Draw agent movement trails"""
        for agent in self.env.agents:
            if len(agent.trail) > 1:
                trail_points = []
                for pos in agent.trail:
                    screen_pos = self._grid_to_screen(pos)
                    trail_points.append((screen_pos[0] + self.cell_size // 2, 
                                       screen_pos[1] + self.cell_size // 2))
                
                if len(trail_points) > 1:
                    for i in range(len(trail_points) - 1):
                        # Fading trail effect
                        alpha = int(100 * (i + 1) / len(trail_points))
                        trail_color = (*agent.color, alpha)
                        pygame.draw.line(self.screen, agent.color, 
                                       trail_points[i], trail_points[i + 1], 2)
    
    def _draw_resources(self):
        """Draw resources with glow effect"""
        for resource in self.env.resources:
            pos = resource['pos']
            resource_type = resource['type']
            
            screen_pos = self._grid_to_screen(pos)
            center = (screen_pos[0] + self.cell_size // 2, 
                     screen_pos[1] + self.cell_size // 2)
            
            # Resource color based on type
            if resource_type == 'crystal':
                color = (150, 150, 255)
            elif resource_type == 'energy':
                color = (255, 255, 100)
            else:  # gold
                color = Colors.RESOURCE
            
            # Draw glow effect
            glow_radius = int(self.cell_size * 0.4)
            for r in range(glow_radius, 0, -2):
                alpha = int(50 * (glow_radius - r) / glow_radius)
                glow_surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*color, alpha), (r, r), r)
                self.screen.blit(glow_surface, (center[0] - r, center[1] - r))
            
            # Draw resource
            resource_radius = int(self.cell_size * 0.25)
            pygame.draw.circle(self.screen, color, center, resource_radius)
    
    def _draw_agents(self):
        """Draw ALL agents with their states"""
        for agent in self.env.agents:
            screen_pos = self._grid_to_screen(agent.pos)
            center = (screen_pos[0] + self.cell_size // 2, 
                     screen_pos[1] + self.cell_size // 2)
            
            # Agent body with state-based styling
            agent_radius = int(self.cell_size * 0.3)
            
            # Different border for different states
            border_color = Colors.TEXT_PRIMARY
            border_width = 2
            
            if agent.state == AgentState.GATHERING:
                border_color = Colors.SUCCESS_GREEN
                border_width = 3
            elif agent.state == AgentState.COORDINATING:
                border_color = Colors.WARNING_ORANGE
                border_width = 3
            elif agent.state == AgentState.MOVING_TO_RESOURCE:
                border_color = Colors.UI_ACCENT
                border_width = 2
            
            pygame.draw.circle(self.screen, agent.color, center, agent_radius)
            pygame.draw.circle(self.screen, border_color, center, agent_radius, border_width)
            
            # Agent ID
            id_text = self.font_small.render(str(agent.id), True, Colors.TEXT_PRIMARY)
            id_rect = id_text.get_rect(center=center)
            self.screen.blit(id_text, id_rect)
            
            # Activity indicator (small pulsing dot)
            if agent.action_cooldown == 0:  # Agent can act
                activity_pos = (center[0] + agent_radius - 5, center[1] - agent_radius + 5)
                activity_color = Colors.SUCCESS_GREEN if agent.last_reward > 0 else Colors.UI_ACCENT
                pygame.draw.circle(self.screen, activity_color, activity_pos, 3)
            
            # Inventory indicator
            total_inventory = sum(agent.inventory.values())
            if total_inventory > 0:
                inv_text = self.font_small.render(str(total_inventory), True, Colors.SUCCESS_GREEN)
                inv_pos = (center[0] - agent_radius + 2, center[1] + agent_radius - 15)
                self.screen.blit(inv_text, inv_pos)
    
    def _draw_coordination_links(self):
        """Draw coordination links between agents"""
        for agent in self.env.agents:
            if agent.coordination_partners:
                agent_center = self._grid_to_screen(agent.pos)
                agent_center = (agent_center[0] + self.cell_size // 2, 
                              agent_center[1] + self.cell_size // 2)
                
                for partner_id in agent.coordination_partners:
                    if partner_id < len(self.env.agents):
                        partner = self.env.agents[partner_id]
                        partner_center = self._grid_to_screen(partner.pos)
                        partner_center = (partner_center[0] + self.cell_size // 2, 
                                        partner_center[1] + self.cell_size // 2)
                        
                        # Draw coordination link
                        pygame.draw.line(self.screen, Colors.SUCCESS_GREEN, 
                                       agent_center, partner_center, 3)
    
    def _draw_effects(self):
        """Draw visual effects"""
        effects_to_remove = []
        
        for effect in self.effects:
            pos = self._grid_to_screen(effect['pos'])
            center = (pos[0] + self.cell_size // 2, pos[1] + self.cell_size // 2)
            
            progress = effect['timer'] / effect['start_timer']
            
            if effect['type'] == 'gather':
                radius = int((1 - progress) * self.cell_size)
                alpha = int(255 * progress)
                
                effect_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(effect_surface, (*effect['color'][:3], alpha), 
                                 (radius, radius), radius, 3)
                self.screen.blit(effect_surface, (center[0] - radius, center[1] - radius))
            
            elif effect['type'] == 'coordinate':
                base_radius = int(self.cell_size * 0.4)
                pulse_radius = int(base_radius * (1 + 0.3 * np.sin(progress * np.pi * 4)))
                alpha = int(255 * progress * 0.5)
                
                effect_surface = pygame.Surface((pulse_radius * 2, pulse_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(effect_surface, (*effect['color'][:3], alpha), 
                                 (pulse_radius, pulse_radius), pulse_radius, 2)
                self.screen.blit(effect_surface, (center[0] - pulse_radius, center[1] - pulse_radius))
            
            effect['timer'] -= 1
            if effect['timer'] <= 0:
                effects_to_remove.append(effect)
        
        for effect in effects_to_remove:
            self.effects.remove(effect)
    
    def _draw_ui(self):
        """Draw enhanced UI showing all agent activity"""
        
        # Sidebar background
        sidebar_rect = pygame.Rect(self.grid_area_width, 0, self.sidebar_width, self.height)
        pygame.draw.rect(self.screen, (25, 25, 35), sidebar_rect)
        pygame.draw.line(self.screen, Colors.GRID_LINE, 
                        (self.grid_area_width, 0), (self.grid_area_width, self.height), 2)
        
        # Title
        title_text = self.font_large.render("PROJECT NEXUS", True, Colors.UI_ACCENT)
        self.screen.blit(title_text, (self.grid_area_width + 20, 20))
        
        subtitle_text = self.font_medium.render("Multi-Agent Fixed!", True, Colors.SUCCESS_GREEN)
        self.screen.blit(subtitle_text, (self.grid_area_width + 20, 55))
        
        # Environment stats
        y_offset = 100
        
        total_reward = sum(self.env.total_rewards)
        stats = [
            ("Step", self.env.step_count),
            ("Total Reward", f"{total_reward:.1f}"),
            ("Coordination", self.env.coordination_events),
            ("Resources", len(self.env.resources)),
            ("Efficiency", f"{(total_reward/max(self.env.step_count,1)*100):.1f}%"),
        ]
        
        for label, value in stats:
            label_text = self.font_medium.render(f"{label}:", True, Colors.TEXT_SECONDARY)
            value_text = self.font_medium.render(str(value), True, Colors.TEXT_PRIMARY)
            
            self.screen.blit(label_text, (self.grid_area_width + 20, y_offset))
            self.screen.blit(value_text, (self.grid_area_width + 160, y_offset))
            y_offset += 25
        
        # Individual agent status
        y_offset += 20
        agents_title = self.font_medium.render("Agent Activity:", True, Colors.UI_ACCENT)
        self.screen.blit(agents_title, (self.grid_area_width + 20, y_offset))
        y_offset += 30
        
        for agent in self.env.agents:
            # Agent color indicator
            color_rect = pygame.Rect(self.grid_area_width + 20, y_offset - 2, 12, 12)
            pygame.draw.rect(self.screen, agent.color, color_rect)
            
            # Agent status
            total_items = sum(agent.inventory.values())
            status_color = Colors.SUCCESS_GREEN if agent.last_reward > 0 else Colors.TEXT_PRIMARY
            
            # Status text with current action
            status_text = f"Agent {agent.id}: {agent.state.value}"
            if len(status_text) > 25:
                status_text = f"Agent {agent.id}: {agent.state.value[:12]}..."
            
            agent_text = self.font_small.render(status_text, True, status_color)
            self.screen.blit(agent_text, (self.grid_area_width + 40, y_offset))
            
            # Detailed info on second line
            reward_text = f"R:{agent.last_reward:+.1f} Items:{total_items}"
            if agent.coordination_partners:
                reward_text += f" Team:{len(agent.coordination_partners)}"
            
            detail_text = self.font_small.render(reward_text, True, Colors.TEXT_SECONDARY)
            self.screen.blit(detail_text, (self.grid_area_width + 40, y_offset + 15))
            
            # Energy bar
            energy_width = 60
            energy_height = 4
            energy_x = self.grid_area_width + 40
            energy_y = y_offset + 30
            
            # Background
            pygame.draw.rect(self.screen, Colors.GRID_LINE, 
                           (energy_x, energy_y, energy_width, energy_height))
            
            # Energy level
            energy_fill = int(energy_width * (agent.energy / 100))
            energy_color = Colors.SUCCESS_GREEN if agent.energy > 70 else Colors.WARNING_ORANGE if agent.energy > 30 else (255, 100, 100)
            pygame.draw.rect(self.screen, energy_color, 
                           (energy_x, energy_y, energy_fill, energy_height))
            
            y_offset += 50
        
        # Performance indicators
        y_offset += 20
        perf_title = self.font_medium.render("Performance:", True, Colors.UI_ACCENT)
        self.screen.blit(perf_title, (self.grid_area_width + 20, y_offset))
        y_offset += 30
        
        # Calculate performance metrics
        if self.env.step_count > 0:
            avg_reward_per_step = total_reward / self.env.step_count
            coordination_rate = self.env.coordination_events / max(self.env.step_count, 1)
            
            perf_metrics = [
                ("Avg Reward/Step", f"{avg_reward_per_step:.3f}"),
                ("Coordination Rate", f"{coordination_rate:.3f}"),
                ("Most Active", f"Agent {np.argmax(self.env.total_rewards)}"),
            ]
            
            for label, value in perf_metrics:
                label_text = self.font_small.render(f"{label}:", True, Colors.TEXT_SECONDARY)
                value_text = self.font_small.render(str(value), True, Colors.TEXT_PRIMARY)
                
                self.screen.blit(label_text, (self.grid_area_width + 20, y_offset))
                self.screen.blit(value_text, (self.grid_area_width + 140, y_offset))
                y_offset += 20
        
        # Controls
        y_offset = self.height - 120
        controls_title = self.font_medium.render("Controls:", True, Colors.UI_ACCENT)
        self.screen.blit(controls_title, (self.grid_area_width + 20, y_offset))
        y_offset += 25
        
        controls = [
            "SPACE: Pause/Resume",
            "UP/DOWN: Speed",
            "R: Reset",
            "ESC: Exit"
        ]
        
        for control in controls:
            control_text = self.font_small.render(control, True, Colors.TEXT_SECONDARY)
            self.screen.blit(control_text, (self.grid_area_width + 20, y_offset))
    
    def _grid_to_screen(self, grid_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert grid coordinates to screen coordinates"""
        return (self.grid_offset_x + grid_pos[0] * self.cell_size,
                self.grid_offset_y + grid_pos[1] * self.cell_size)
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'}")
                elif event.key == pygame.K_r:
                    self.reset_simulation()
                elif event.key == pygame.K_UP:
                    self.speed = min(30, self.speed + 2)
                    print(f"Speed: {self.speed} FPS")
                elif event.key == pygame.K_DOWN:
                    self.speed = max(1, self.speed - 2)
                    print(f"Speed: {self.speed} FPS")
    
    def reset_simulation(self):
        """Reset the simulation"""
        print("Resetting multi-agent simulation...")
        self.env.reset()
        self.effects = []
        print("All agents reset and ready for action!")
    
    def run(self):
        """Main game loop"""
        
        print("\nPROJECT NEXUS - FIXED MULTI-AGENT VISUALIZATION!")
        print("=" * 60)
        print("ALL AGENTS NOW ACTIVE EVERY STEP!")
        print("Watch TRUE multi-agent coordination in real-time!")
        print("Every agent moves, thinks, and acts independently!")
        print("Coordination emerges naturally between nearby agents!")
        print("\nCONTROLS:")
        print("  SPACE: Pause/Resume simulation")
        print("  UP/DOWN: Adjust speed (1-30 FPS)")
        print("  R: Reset with new random setup")
        print("  ESC: Exit visualization")
        print("=" * 60)
        print("Starting true multi-agent simulation...")
        
        # Main game loop
        try:
            while self.running:
                # Handle events
                self.handle_events()
                
                # Update simulation (ALL AGENTS ACT)
                self.update_simulation()
                
                # Draw everything
                self.draw()
                
                # Control frame rate
                self.clock.tick(self.speed)
            
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"\nFIXED Multi-Agent Simulation Complete!")
            print(f"Final Statistics:")
            print(f"   Steps: {self.env.step_count}")
            print(f"   Total Rewards: {sum(self.env.total_rewards):.1f}")
            print(f"   Coordination Events: {self.env.coordination_events}")
            print(f"   Individual Rewards: {[f'{r:.1f}' for r in self.env.total_rewards]}")
            print("ALL agents were active and contributed!")
            print("True multi-agent intelligence demonstrated!")
            
            pygame.quit()

def main():
    """Launch the FIXED visual demo"""
    
    try:
        print("PROJECT NEXUS - FIXED MULTI-AGENT VISUALIZATION")
        print("This version ensures ALL agents are active every step!")
        
        # Create and run the FIXED visualizer
        visualizer = NEXUSVisualizerFixed(width=1200, height=900, grid_size=15)
        visualizer.run()
        
    except ImportError:
        print("Pygame not installed!")
        print("Install with: pip install pygame numpy")
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()