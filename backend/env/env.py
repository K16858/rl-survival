import numpy as np
import pygame
from enum import IntEnum
from env.tile import TileType
import random
from typing import Tuple, Dict, Any, List
import sys

class IslandEnvironment:
    def __init__(self, size=500, seed=None, tile_size=3, fps=30, movement_type="continuous", window_size=600):
        """
        Initialize the deserted island environment
        
        Args:
            size: Map size (size x size)
            seed: Random seed
            tile_size: Tile size in pixels for rendering
            fps: Frame rate
            movement_type: "grid" or "continuous"
            window_size: Maximum window size in pixels
        """
        self.size = size
        self.tile_size = tile_size
        self.fps = fps
        self.screen = None
        self.clock = None
        self.is_initialized = False
        self.movement_type = "continuous"  # Always use continuous movement
        self.window_size = window_size
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # Initialize map
        self.map = np.zeros((size, size), dtype=int)
        
        # Agent's initial position (supports continuous values)
        self.agent_pos = None
        
        # Define action space (8 directions for better continuous movement)
        self.action_space = [
            (0, 1),    # Right
            (1, 1),    # Down-Right
            (1, 0),    # Down
            (1, -1),   # Down-Left
            (0, -1),   # Left
            (-1, -1),  # Up-Left
            (-1, 0),   # Up
            (-1, 1),   # Up-Right
        ]
            
        # Movement speed
        self.move_speed = 0.5  # Movement per step
        
        # Generate the map
        self._generate_map()
        
    def _generate_map(self):
        """Generate the island map"""
        # Use Perlin noise for natural terrain generation
        from noise import pnoise2
        
        # Generate basic height map
        height_map = np.zeros((self.size, self.size))
        scale = 0.01  # Noise scale
        
        for i in range(self.size):
            for j in range(self.size):
                # Decrease height based on distance from center
                dist_from_center = np.sqrt((i - self.size/2)**2 + (j - self.size/2)**2)
                island_factor = max(0, 1 - dist_from_center / (self.size * 0.4))
                
                # Generate noise for terrain variation
                noise_val = pnoise2(i * scale, j * scale, octaves=6, persistence=0.5)
                
                # Update height map
                height_map[i, j] = (noise_val * 0.5 + 0.5) * island_factor
        
        # Determine tile types based on height
        for i in range(self.size):
            for j in range(self.size):
                height = height_map[i, j]
                
                if height < 0.2:
                    self.map[i, j] = TileType.OCEAN
                elif height < 0.3:
                    self.map[i, j] = TileType.BEACH
                elif height < 0.7:
                    self.map[i, j] = TileType.GRASS
                else:
                    # Higher elevation is mountains
                    self.map[i, j] = TileType.MOUNTAIN
        
        # Generate rivers
        self._generate_rivers(height_map)
        
        # Set agent's initial position to grass area
        valid_positions = [(i, j) for i in range(self.size) for j in range(self.size) 
                          if self.map[i, j] == TileType.GRASS]
        if valid_positions:
            i, j = random.choice(valid_positions)
            # Place in center of the tile for continuous movement
            self.agent_pos = (i + 0.5, j + 0.5)
    
    def _generate_rivers(self, height_map):
        """Generate rivers based on the height map"""
        # Force generate at least 2 rivers
        num_rivers = max(2, random.randint(1, 5))
        print(f"Generating {num_rivers} rivers...")
        
        rivers_created = 0
        max_attempts = 20  # Maximum number of attempts for river generation
        
        for attempt in range(max_attempts):
            if rivers_created >= num_rivers:
                break
                
            # Find river source points from higher elevations
            # Increase candidates by lowering the threshold
            height_threshold = 0.6 - (attempt * 0.05)  # Lower threshold with each attempt
            start_candidates = [(i, j) for i in range(self.size) for j in range(self.size)
                               if self.map[i, j] == TileType.GRASS and height_map[i, j] > max(0.3, height_threshold)]
            
            if not start_candidates:
                # Ignore height condition, start from any grass tile
                start_candidates = [(i, j) for i in range(self.size) for j in range(self.size)
                                  if self.map[i, j] == TileType.GRASS]
                                  
            if not start_candidates:
                # If still not found, start from any land tile
                start_candidates = [(i, j) for i in range(self.size) for j in range(self.size)
                                  if self.map[i, j] != TileType.OCEAN]
                
            if not start_candidates:
                print("Warning: Could not find starting point for river")
                continue
                
            # Randomly select starting point
            current = random.choice(start_candidates)
            river_path = [current]
            river_length = 0
            max_river_length = self.size // 2  # Maximum river length
            
            # Flow river downward
            while river_length < max_river_length:
                i, j = current
                neighbors = []
                
                # Check adjacent cells (including diagonal directions)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                            
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            # Prioritize horizontal or downward movement (weighted)
                            flow_weight = 0.0
                            if di >= 0:  # Tendency to flow downward or horizontally
                                flow_weight -= 0.1
                                
                            # Consider previous direction to favor straight flow (if exists)
                            if len(river_path) > 1:
                                prev_i, prev_j = river_path[-2]
                                if (di == i - prev_i and dj == j - prev_j):  # Same direction
                                    flow_weight -= 0.2
                                    
                            # Flow toward lower elevations
                            if ni < self.size and nj < self.size:
                                neighbors.append((ni, nj, height_map[ni, nj] + flow_weight))
                
                # Find the lowest cell (prioritize existing rivers)
                neighbors.sort(key=lambda x: x[2])  # Sort by height
                
                # Find next point
                next_pos_found = False
                for ni, nj, _ in neighbors:
                    if (ni, nj) not in river_path:  # Don't merge with existing river path
                        if self.map[ni, nj] != TileType.OCEAN:  # Don't flow into ocean
                            current = (ni, nj)
                            river_path.append(current)
                            river_length += 1
                            next_pos_found = True
                            break
                
                # End if no next point found
                if not next_pos_found or self.map[i, j] == TileType.OCEAN:
                    break
            
            # Apply river if length is sufficient
            min_river_length = 5  # Minimum river length
            if river_length >= min_river_length:
                # Apply river to map (with width)
                for i, j in river_path:
                    if self.map[i, j] != TileType.OCEAN:
                        self.map[i, j] = TileType.RIVER
                        
                        # 20% chance to increase river width
                        if random.random() < 0.2:
                            di, dj = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.size and 0 <= nj < self.size:
                                if self.map[ni, nj] != TileType.OCEAN and self.map[ni, nj] != TileType.RIVER:
                                    self.map[ni, nj] = TileType.RIVER
                
                rivers_created += 1
                print(f"Created river #{rivers_created} - {river_length} tiles")
        
        # If still not enough rivers, force create one
        if rivers_created == 0:
            print("Failed to naturally generate rivers. Forcing river creation...")
            self._force_create_river()
    
    def _force_create_river(self):
        """Force create a river (fallback for worst case)"""
        # Create a straight river near the center of the island
        center = self.size // 2
        width = self.size // 20  # River width
        
        start_i = center - self.size // 4
        end_i = center + self.size // 4
        
        for i in range(start_i, end_i):
            for j in range(center - width, center + width):
                if 0 <= i < self.size and 0 <= j < self.size:
                    if self.map[i, j] != TileType.OCEAN:
                        self.map[i, j] = TileType.RIVER
        
        print(f"Forcibly created river - {(end_i - start_i) * (width * 2)} tiles")

    def _initialize_pygame(self):
        """Initialize PyGame"""
        pygame.init()
        
        # Calculate display size
        display_size = min(self.window_size, self.size * self.tile_size)
        self.display_size = display_size
        self.screen = pygame.display.set_mode((display_size, display_size))
        pygame.display.set_caption("Desert Island Survival Game")
        
        self.clock = pygame.time.Clock()
        self.is_initialized = True
    
    def reset(self):
        """Reset the environment and return the initial state"""
        # Reset agent position to a grass tile
        valid_positions = [(i, j) for i in range(self.size) for j in range(self.size) 
                          if self.map[i, j] == TileType.GRASS]
        if valid_positions:
            i, j = random.choice(valid_positions)
            # Place in center of tile for continuous movement
            self.agent_pos = (i + 0.5, j + 0.5)
            
        return self._get_observation()
    
    def step(self, action_idx):
        """
        Update environment based on agent's action
        
        Args:
            action_idx: Action index
            
        Returns:
            observation: New state observation
            reward: Reward value
            done: Episode termination flag
            info: Debug information
        """
        if not 0 <= action_idx < len(self.action_space):
            raise ValueError(f"Invalid action: {action_idx}")
        
        # Get current position
        current_i, current_j = self.agent_pos
        
        # Get direction from action
        di, dj = self.action_space[action_idx]
        
        # For diagonal movement, normalize the distance
        if di != 0 and dj != 0:
            # Diagonal directions have √2 distance, normalize to unit length
            norm = 1.0 / np.sqrt(di*di + dj*dj)
            di *= norm
            dj *= norm
            
        # Calculate new position with continuous movement
        new_i = current_i + di * self.move_speed
        new_j = current_j + dj * self.move_speed
        
        # Ensure within map bounds
        new_i = max(0.0, min(self.size - 0.001, new_i))
        new_j = max(0.0, min(self.size - 0.001, new_j))
        
        # Check tile type at the new position
        tile_type = self.get_tile_at_position(new_i, new_j)
        
        # Only allow movement to non-ocean tiles
        if tile_type != TileType.OCEAN:
            self.agent_pos = (new_i, new_j)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check for episode termination (not used in this example)
        done = False
        
        return self._get_observation(), reward, done, {}
    
    def get_tile_at_position(self, i, j):
        """Get tile type at continuous position
        
        Args:
            i, j: Continuous position coordinates
            
        Returns:
            TileType: Type of tile at the position
        """
        # Convert floating point to integer coordinates
        map_i, map_j = int(i), int(j)
        
        # Check if within bounds
        if 0 <= map_i < self.size and 0 <= map_j < self.size:
            return self.map[map_i, map_j]
        else:
            return TileType.OCEAN  # Out of bounds is ocean
    
    def _get_observation(self):
        """Return the agent's observation"""
        # Get integer position from continuous position
        i, j = map(int, self.agent_pos)
        
        # Create vision array (11x11 area around agent)
        vision_size = 11
        half_vision = vision_size // 2
        
        vision = np.zeros((vision_size, vision_size), dtype=int)
        
        for vi in range(vision_size):
            for vj in range(vision_size):
                map_i = i + (vi - half_vision)
                map_j = j + (vj - half_vision)
                
                if 0 <= map_i < self.size and 0 <= map_j < self.size:
                    vision[vi, vj] = self.map[map_i, map_j]
                else:
                    vision[vi, vj] = TileType.OCEAN  # Outside map is ocean
        
        return {
            'position': self.agent_pos,  # Continuous position
            'vision': vision,
            'current_tile': self.get_tile_at_position(*self.agent_pos)
        }
    
    def _calculate_reward(self):
        """Simple reward function example"""
        # Get current tile type
        current_tile = self.get_tile_at_position(*self.agent_pos)
        
        # Return reward based on tile type
        if current_tile == TileType.BEACH:
            return 0.5  # Beach reward
        elif current_tile == TileType.GRASS:
            return 1.0  # Grass reward
        elif current_tile == TileType.RIVER:
            return 0.7  # River reward
        else:
            return 0.0  # Other areas
    
    def render(self, mode='human'):
        """Visualize the environment"""
        if not self.is_initialized:
            self._initialize_pygame()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
                
        self.screen.fill((0, 0, 0))  # Fill with black
        
        # Calculate view area centered on agent
        i_center, j_center = self.agent_pos
        view_radius = self.display_size // (2 * self.tile_size)
        
        # Calculate display bounds
        i_min = max(0, int(i_center - view_radius))
        i_max = min(self.size, int(i_center + view_radius + 1))
        j_min = max(0, int(j_center - view_radius))
        j_max = min(self.size, int(j_center + view_radius + 1))
        
        # Draw tiles within view
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                # Convert to screen coordinates
                screen_i = int((i - (i_center - view_radius)) * self.tile_size)
                screen_j = int((j - (j_center - view_radius)) * self.tile_size)
                
                # Draw if on screen
                if 0 <= screen_i < self.display_size and 0 <= screen_j < self.display_size:
                    # Draw the tile
                    color = TileType.get_color(self.map[i, j])
                    pygame.draw.rect(
                        self.screen, 
                        color, 
                        (screen_j, screen_i, self.tile_size, self.tile_size)
                    )
        
        # Draw agent (red circle)
        agent_screen_i = int((i_center - (i_center - view_radius)) * self.tile_size)
        agent_screen_j = int((j_center - (j_center - view_radius)) * self.tile_size)
        
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (agent_screen_j, agent_screen_i),
            max(1, self.tile_size // 2)
        )
        
        # Draw minimap in bottom right corner
        minimap_size = self.display_size // 4
        minimap_surface = pygame.Surface((minimap_size, minimap_size))
        minimap_tile_size = max(1, minimap_size // self.size)
        
        for i in range(0, self.size, max(1, self.size // (minimap_size // minimap_tile_size))):
            for j in range(0, self.size, max(1, self.size // (minimap_size // minimap_tile_size))):
                mini_i = i * minimap_tile_size // max(1, self.size // (minimap_size // minimap_tile_size))
                mini_j = j * minimap_tile_size // max(1, self.size // (minimap_size // minimap_tile_size))
                
                if mini_i < minimap_size and mini_j < minimap_size:
                    color = TileType.get_color(self.map[i, j])
                    pygame.draw.rect(
                        minimap_surface,
                        color,
                        (mini_j, mini_i, minimap_tile_size, minimap_tile_size)
                    )
        
        # Draw agent position on minimap
        agent_mini_i = int(self.agent_pos[0] * minimap_tile_size / max(1, self.size // (minimap_size // minimap_tile_size)))
        agent_mini_j = int(self.agent_pos[1] * minimap_tile_size / max(1, self.size // (minimap_size // minimap_tile_size)))
        if 0 <= agent_mini_i < minimap_size and 0 <= agent_mini_j < minimap_size:
            pygame.draw.rect(
                minimap_surface,
                (255, 0, 0),  # Red
                (agent_mini_j, agent_mini_i, minimap_tile_size, minimap_tile_size)
            )
        
        # Position minimap in bottom right
        self.screen.blit(minimap_surface, (self.display_size - minimap_size - 10, self.display_size - minimap_size - 10))
        
        # Update display:
        pygame.display.flip()
        self.clock.tick(self.fps)
           
    def close(self):
        """Close the environment"""
        if self.is_initialized:
            pygame.quit()
            self.is_initialized = False
    
    def find_nearest_tile(self, start_pos: Tuple[float, float], tile_type: TileType, max_distance: int = 20) -> Tuple[int, int]:
        """Find the nearest tile of specified type
        
        Args:
            start_pos: Starting position (i, j) - can be continuous
            tile_type: Type of tile to search for
            max_distance: Maximum search distance
            
        Returns:
            Tuple[int, int]: Position of nearest tile or None if not found
        """
        # Convert continuous position to integer coordinates
        i_start, j_start = int(start_pos[0]), int(start_pos[1])
        
        # Use BFS to find nearest tile
        queue = [(i_start, j_start, 0)]  # (i, j, distance)
        visited = set([(i_start, j_start)])
        
        while queue:
            i, j, distance = queue.pop(0)
            
            # Check if this is the target tile
            if 0 <= i < self.size and 0 <= j < self.size and self.map[i, j] == tile_type:
                return (i, j)
                
            # Stop if max distance reached
            if distance >= max_distance:
                continue
                
            # Add neighbors to queue
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Use 4-connectivity for pathfinding
                ni, nj = i + di, j + dj
                
                # Check if in bounds
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    # Don't visit ocean tiles or already visited tiles
                    if (ni, nj) not in visited and self.map[ni, nj] != TileType.OCEAN:
                        visited.add((ni, nj))
                        queue.append((ni, nj, distance + 1))
        
        # If no tile found
        return None
    
    def move_agent_to(self, target_pos: Tuple[int, int]) -> List[int]:
        """Move agent to target position using pathfinding
        
        Args:
            target_pos: Target position to move to (integer coordinates)
            
        Returns:
            List[int]: List of actions taken
        """
        if target_pos is None:
            return []
            
        i_target, j_target = target_pos
        actions_taken = []
        
        # Maximum steps to prevent infinite loops
        max_steps = 200
        steps = 0
        
        while steps < max_steps:
            # Get current integer position from continuous position
            i_current, j_current = map(int, self.agent_pos)
            
            # Check if we've reached the target
            if (i_current, j_current) == (i_target, j_target):
                break
                
            # Find best direction to move
            best_action = None
            best_distance = float('inf')
            
            for action_idx, (di, dj) in enumerate(self.action_space):
                # For diagonal movement, normalize
                if di != 0 and dj != 0:
                    norm = 1.0 / np.sqrt(di*di + dj*dj)
                    di_norm, dj_norm = di * norm, dj * norm
                    ni, nj = self.agent_pos[0] + di_norm * self.move_speed, self.agent_pos[1] + dj_norm * self.move_speed
                else:
                    ni, nj = self.agent_pos[0] + di * self.move_speed, self.agent_pos[1] + dj * self.move_speed
                
                # Convert to integer for grid checking
                ni_int, nj_int = int(ni), int(nj)
                
                # Check if valid move
                if 0 <= ni_int < self.size and 0 <= nj_int < self.size and self.map[ni_int, nj_int] != TileType.OCEAN:
                    # Calculate Manhattan distance to target
                    distance = abs(ni_int - i_target) + abs(nj_int - j_target)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_action = action_idx
            
            if best_action is not None:
                # Take the action
                observation, reward, done, info = self.step(best_action)
                actions_taken.append(best_action)
                
                # Check if we're stuck (making no progress)
                new_i_int, new_j_int = map(int, self.agent_pos)
                if (new_i_int, new_j_int) == (i_current, j_current) and steps > 10:
                    # Try to move to center of current tile to unstuck
                    self.agent_pos = (new_i_int + 0.5, new_j_int + 0.5)
            else:
                # No valid move found
                break
                
            steps += 1
            
        return actions_taken
