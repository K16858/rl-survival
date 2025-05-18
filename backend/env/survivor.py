from enum import Enum
from typing import Dict, List, Any, Tuple
import numpy as np
import random
from item import Item, ItemType
from env.tile import TileType

class Survivor:
    """Survivor class - represents a person surviving on a deserted island"""
    
    def __init__(self, name: str = "A", environment=None):
        """Initialize the survivor"""
        # All stats are in 0-1 range
        self.name = name
        self.health = 1.0       # Health: 0 means death
        self.satiety = 1.0      # Satiety: 0 means starving, 1 means full
        self.hydration = 1.0    # Hydration: 0 means dehydrated, 1 means well-hydrated
        self.temperature = 0.5  # Body temperature: 0 means hypothermia, 1 means hyperthermia
        self.stamina = 1.0      # Stamina: 0 means exhausted
        self.rest = 1.0         # Rest level: 0 means exhausted, 1 means well-rested
        self.stress = 0.0       # Stress: 1 means maximum stress
        
        # Reference to environment
        self.environment = environment
        
        # Inventory
        self.inventory: List[Item] = []
        self.max_inventory_size = 10  # Maximum number of items
        self.max_weight = 20.0  # Maximum weight capacity (kg)
        
        # Status flags
        self.is_alive = True
        self.days_survived = 0

        # Movement parameters for continuous movement
        self.move_speed = 0.5

    def update(self, time_delta: float = 1.0):
        """Update stats based on time passage
        
        Args:
            time_delta: Time passed (in-game hours, default is 1 hour)
        """
        # Decrease satiety over time
        self.satiety -= 0.05 * time_delta
        
        # Decrease hydration over time
        self.hydration -= 0.08 * time_delta
        
        # Decrease rest level over time
        self.rest -= 0.04 * time_delta
        
        # Natural stamina recovery (slower when rest is low)
        stamina_recovery = 0.02 * time_delta * self.rest * 0.5
        self.stamina = min(1.0, self.stamina + stamina_recovery)
        
        # Health decrease due to hunger or thirst
        if self.satiety < 0.2:
            self.health -= 0.02 * time_delta
        if self.hydration < 0.2:
            self.health -= 0.05 * time_delta
            
        # Effects of extreme temperature
        if self.temperature < 0.2 or self.temperature > 0.8:
            self.health -= 0.03 * time_delta
            self.stress += 0.05 * time_delta
            
        # Effects of low rest
        if self.rest < 0.3:
            self.stamina -= 0.02 * time_delta
            self.stress += 0.04 * time_delta
            
        # Clamp all stats to valid range
        self._clamp_stats()
        
        # Check if still alive
        if self.health <= 0:
            self.is_alive = False
            
    def _clamp_stats(self):
        """Clamp all stats to 0-1 range"""
        self.health = max(0.0, min(1.0, self.health))
        self.satiety = max(0.0, min(1.0, self.satiety))
        self.hydration = max(0.0, min(1.0, self.hydration))
        self.temperature = max(0.0, min(1.0, self.temperature))
        self.stamina = max(0.0, min(1.0, self.stamina))
        self.rest = max(0.0, min(1.0, self.rest))
        self.stress = max(0.0, min(1.0, self.stress))
            
    def eat(self, item_index: int = None) -> bool:
        """Eat food
        
        Args:
            item_index: Index of item in inventory (None for auto-selection)
            
        Returns:
            bool: Success or failure
        """
        # Auto-select item if not specified
        if item_index is None:
            # List all food items
            food_items = []
            for i, item in enumerate(self.inventory):
                if item.item_type in [ItemType.FRUIT, ItemType.FISH, ItemType.MEAT]:
                    # Estimate satiety effect based on food type
                    satiety_effect = 0.2
                    if item.item_type == ItemType.FISH:
                        satiety_effect = 0.4
                    elif item.item_type == ItemType.MEAT:
                        satiety_effect = 0.6
                    
                    # Consider additional effects
                    extra_effect = sum(item.effects.get(stat, 0) for stat in ["health", "stamina"])
                    
                    # Total value
                    value = satiety_effect + extra_effect
                    food_items.append((i, value))
            
            if not food_items:
                return False  # No food available
                
            # Select item with highest value
            item_index, _ = max(food_items, key=lambda x: x[1])
        
        # Validate the item index
        if not self._check_item_index(item_index):
            return False
            
        item = self.inventory[item_index]
        if item.item_type not in [ItemType.FRUIT, ItemType.FISH, ItemType.MEAT]:
            return False
            
        # Apply effects from using the item
        effects = item.use()
        for stat, value in effects.items():
            if hasattr(self, stat):
                current = getattr(self, stat)
                setattr(self, stat, max(0.0, min(1.0, current + value)))
                
        # Increase satiety based on food type
        if item.item_type == ItemType.FRUIT:
            self.satiety += 0.2
        elif item.item_type == ItemType.FISH:
            self.satiety += 0.4
        elif item.item_type == ItemType.MEAT:
            self.satiety += 0.6
            
        # Remove consumed item if durability is zero
        if item.durability <= 0:
            self.inventory.pop(item_index)
            
        self._clamp_stats()
        return True
    
    def drink(self, item_index: int = None) -> bool:
        """Drink water
        
        Args:
            item_index: Index of item in inventory (None for auto-selection)
            
        Returns:
            bool: Success or failure
        """
        # First try to use water items from inventory
        if item_index is not None or any(item.item_type == ItemType.WATER for item in self.inventory):
            # Auto-select item if not specified
            if item_index is None:
                # List all drink items
                drink_items = []
                for i, item in enumerate(self.inventory):
                    if item.item_type == ItemType.WATER:
                        # Consider additional effects
                        extra_effect = sum(item.effects.get(stat, 0) for stat in ["health", "stamina"])
                        
                        # Total value
                        value = 0.5 + extra_effect  # Base hydration effect plus extras
                        drink_items.append((i, value))
                
                # Select item with highest value
                item_index, _ = max(drink_items, key=lambda x: x[1])
            
            # Validate the item index
            if not self._check_item_index(item_index):
                return False
                
            item = self.inventory[item_index]
            if item.item_type != ItemType.WATER:
                return False
                
            # Apply effects from using the item
            effects = item.use()
            for stat, value in effects.items():
                if hasattr(self, stat):
                    current = getattr(self, stat)
                    setattr(self, stat, max(0.0, min(1.0, current + value)))
                    
            # Increase hydration
            self.hydration += 0.5
                
            # Remove consumed item if durability is zero
            if item.durability <= 0:
                self.inventory.pop(item_index)
                
            self._clamp_stats()
            return True
        
        # If no water items in inventory, find the nearest river
        if self.environment:
            from env.env import TileType  # Modified import path to avoid circular import
            
            # Find nearest river tile
            river_pos = self.environment.find_nearest_tile(
                self.environment.agent_pos, 
                TileType.RIVER
            )
            
            if river_pos:
                # Move to the river
                self.environment.move_agent_to(river_pos)
                
                # Check if we reached a river tile
                # For continuous movement, check the tile type at current position
                current_tile = self.environment.get_tile_at_position(*self.environment.agent_pos)
                
                if current_tile == TileType.RIVER:
                    # Drink from river - less hydration than water bottle, slight risk
                    self.hydration += 0.4
                    
                    # Small chance of getting sick from river water
                    if random.random() < 0.1:
                        self.health -= 0.05
                        
                    # Consume stamina due to travel
                    self.stamina -= 0.1
                    
                    self._clamp_stats()
                    return True
        
        # If we couldn't drink from inventory or river
        return False
    
    def sleep_action(self, hours: float) -> bool:
        """Sleep to recover
        
        Args:
            hours: Sleep duration (in-game hours)
            
        Returns:
            bool: Success or failure
        """
        if self.stress > 0.8:  # Too stressed to sleep
            return False
            
        # Recovery effects from sleep
        self.rest += 0.1 * hours
        self.stamina += 0.1 * hours
        self.stress -= 0.05 * hours
        
        # Natural decrease in satiety and hydration during sleep
        self.satiety -= 0.02 * hours
        self.hydration -= 0.01 * hours
        
        self._clamp_stats()
        return True
        
    def add_item(self, item: Item) -> bool:
        """Add item to inventory
        
        Args:
            item: Item to add
            
        Returns:
            bool: Success or failure
        """
        # Check inventory capacity
        if len(self.inventory) >= self.max_inventory_size:
            return False
            
        # Check weight limit
        current_weight = sum(item.weight for item in self.inventory)
        if current_weight + item.weight > self.max_weight:
            return False
            
        self.inventory.append(item)
        return True
        
    def remove_item(self, item_index: int) -> Item:
        """Remove item from inventory
        
        Args:
            item_index: Index of item in inventory
            
        Returns:
            Item: Removed item (None if invalid index)
        """
        if not self._check_item_index(item_index):
            return None
            
        return self.inventory.pop(item_index)
    
    def craft_item(self, recipe: Dict[ItemType, int], result_item: Item) -> bool:
        """Craft an item
        
        Args:
            recipe: Materials needed for crafting {ItemType: count}
            result_item: Item to be crafted
            
        Returns:
            bool: Success or failure
        """
        # Check available materials
        available_materials = {}
        for item in self.inventory:
            if item.item_type in recipe:
                available_materials[item.item_type] = available_materials.get(item.item_type, 0) + 1
        
        # Check if we have enough materials
        for item_type, count in recipe.items():
            if available_materials.get(item_type, 0) < count:
                return False
        
        # Check stamina
        if self.stamina < 0.2:
            return False
            
        # Consume materials
        for item_type, count in recipe.items():
            consumed = 0
            for i in range(len(self.inventory) - 1, -1, -1):  # Search from the end
                if consumed >= count:
                    break
                if self.inventory[i].item_type == item_type:
                    self.inventory.pop(i)
                    consumed += 1
        
        # Consume stamina
        self.stamina -= 0.2
        # Reduce stress (satisfaction)
        self.stress = max(0.0, self.stress - 0.1)
        
        # Add result item
        success = self.add_item(result_item)
        if not success:
            # Failed to add item (inventory full)
            return False
            
        self._clamp_stats()
        return True
    
    def heal(self) -> bool:
        """Use healing item (auto-selected)
        
        Returns:
            bool: Success or failure
        """
        # List all healing items
        healing_items = []
        for i, item in enumerate(self.inventory):
            if item.item_type == ItemType.MEDICINE:
                # Estimate health effect
                heal_effect = item.effects.get("health", 0)
                if heal_effect > 0:
                    healing_items.append((i, heal_effect))
        
        if not healing_items:
            return False  # No healing items available
            
        # Select item with highest effect
        best_heal_index, _ = max(healing_items, key=lambda x: x[1])
        
        # Use the medicine
        item = self.inventory[best_heal_index]
        effects = item.use()
        
        # Apply effects
        for stat, value in effects.items():
            if hasattr(self, stat):
                current = getattr(self, stat)
                setattr(self, stat, max(0.0, min(1.0, current + value)))
                
        # Remove used item if durability is zero
        if item.durability <= 0:
            self.inventory.pop(best_heal_index)
            
        self._clamp_stats()
        return True
        
    def auto_eat(self) -> bool:
        """Automatically eat when hungry
        
        Returns:
            bool: Success or failure
        """
        if self.satiety > 0.5:  # Not hungry enough
            return False
            
        return self.eat()  # Auto-select food
    
    def auto_drink(self) -> bool:
        """Automatically drink when thirsty
        
        Returns:
            bool: Success or failure
        """
        if self.hydration > 0.5:  # Not thirsty enough
            return False
            
        return self.drink()  # Auto-select drink
    
    def auto_action(self) -> str:
        """Automatically choose the best action based on current status
        
        Returns:
            str: Description of the action taken
        """
        # Check actions by priority
        
        # 1. Heal when health is critical
        if self.health < 0.3:
            if self.heal():
                return "Used medicine to recover"
                
        # 2. Drink when dehydrated
        if self.hydration < 0.3:
            if self.drink():
                return "Drank water"
                
        # 3. Eat when hungry
        if self.satiety < 0.3:
            if self.eat():
                return "Ate food"
                
        # 4. Rest when tired
        if self.rest < 0.2 or self.stamina < 0.2:
            if self.sleep_action(2.0):  # Rest for 2 hours
                return "Took a rest"
                
        # Default action
        return "Observing the surroundings"
    
    def craft_tool(self) -> bool:
        """Craft a tool from inventory items
        
        Returns:
            bool: Success or failure
        """
        from item import ItemType, Item
        
        # Define some common tool recipes
        recipes = {
            "Stone Axe": {
                "materials": {ItemType.STONE: 2, ItemType.WOOD: 1},
                "result": Item(ItemType.TOOL, "Stone Axe", 2.0, 0.8, {})
            },
            "Fishing Rod": {
                "materials": {ItemType.WOOD: 2, ItemType.FIBER: 1},
                "result": Item(ItemType.TOOL, "Fishing Rod", 1.0, 0.7, {})
            },
            "Torch": {
                "materials": {ItemType.WOOD: 1, ItemType.FIBER: 1},
                "result": Item(ItemType.TOOL, "Torch", 0.5, 0.5, {})
            }
        }
        
        # Check available materials
        available_materials = {}
        for item in self.inventory:
            if item.item_type in [ItemType.STONE, ItemType.WOOD, ItemType.FIBER]:
                available_materials[item.item_type] = available_materials.get(item.item_type, 0) + 1
        
        # Try to craft each tool by priority
        for tool_name, recipe_data in recipes.items():
            materials = recipe_data["materials"]
            result_item = recipe_data["result"]
            
            # Check if we have enough materials
            can_craft = True
            for item_type, count in materials.items():
                if available_materials.get(item_type, 0) < count:
                    can_craft = False
                    break
                    
            if can_craft:
                return self.craft_item(materials, result_item)
        
        return False  # Couldn't craft any tool
    
    def move_to_water_and_drink(self) -> bool:
        """Move to visible water tile and drink
        
        Returns:
            bool: Success or failure
        """
        if not self.environment:
            return False
            
        from env.env import TileType
        
        # Get visible water tiles
        visible_tiles = self.environment.get_visible_tiles(self.environment.agent_pos)
        water_tiles = [pos for pos, tile_type in visible_tiles.items() if tile_type == TileType.RIVER]
        
        if not water_tiles:
            return False  # No water tile visible
            
        # Find nearest water tile
        agent_pos = np.array(self.environment.agent_pos)
        distances = [np.linalg.norm(np.array(pos) - agent_pos) for pos in water_tiles]
        nearest_water = water_tiles[np.argmin(distances)]
        
        # Move to water tile
        self.environment.move_agent_to(nearest_water)
        
        # Consume stamina for movement
        distance = np.linalg.norm(np.array(nearest_water) - agent_pos)
        self.stamina -= 0.05 * distance
        
        # Drink from river
        self.hydration += 0.4
        
        # Small chance of getting sick from river water
        if random.random() < 0.1:
            self.health -= 0.05
            
        self._clamp_stats()
        return True
        
    def collect_nearest_item(self) -> bool:
        """Collect the nearest visible item
        
        Returns:
            bool: Success or failure
        """
        if not self.environment:
            return False
            
        # Get visible items
        visible_items = self.environment.get_visible_items(self.environment.agent_pos)
        
        if not visible_items:
            return False  # No items visible
            
        # Find nearest item
        agent_pos = np.array(self.environment.agent_pos)
        item_positions = [item_info['position'] for item_info in visible_items]
        distances = [np.linalg.norm(np.array(pos) - agent_pos) for pos in item_positions]
        nearest_idx = np.argmin(distances)
        nearest_item_info = visible_items[nearest_idx]
        
        # Move to item position
        self.environment.move_agent_to(nearest_item_info['position'])
        
        # Consume stamina for movement
        distance = np.linalg.norm(np.array(nearest_item_info['position']) - agent_pos)
        self.stamina -= 0.05 * distance
        
        # Try to pick up the item
        item = nearest_item_info['item']
        success = self.add_item(item)
        
        # If successful, remove item from environment
        if success:
            self.environment.remove_item_at(nearest_item_info['position'])
        
        self._clamp_stats()
        return success
    
    def move_to_tent_and_rest(self) -> bool:
        """Move to visible tent and rest
        
        Returns:
            bool: Success or failure
        """
        if not self.environment:
            return False
            
        from env.env import TileType
        
        # Get visible tent tiles
        visible_tiles = self.environment.get_visible_tiles(self.environment.agent_pos)
        tent_tiles = [pos for pos, tile_type in visible_tiles.items() if tile_type == TileType.TENT]
        
        if not tent_tiles:
            return False  # No tent visible
            
        # Find nearest tent
        agent_pos = np.array(self.environment.agent_pos)
        distances = [np.linalg.norm(np.array(pos) - agent_pos) for pos in tent_tiles]
        nearest_tent = tent_tiles[np.argmin(distances)]
        
        # Move to tent
        self.environment.move_agent_to(nearest_tent)
        
        # Consume stamina for movement
        distance = np.linalg.norm(np.array(nearest_tent) - agent_pos)
        self.stamina -= 0.05 * distance
        
        # Rest in tent (better rest efficiency than normal sleep)
        rest_hours = 2.0
        self.rest += 0.15 * rest_hours
        self.stamina += 0.12 * rest_hours
        self.stress -= 0.08 * rest_hours
        
        # Tent provides better temperature regulation
        if self.temperature < 0.4:  # Too cold
            self.temperature += 0.1
        elif self.temperature > 0.6:  # Too hot
            self.temperature -= 0.1
            
        # Natural decrease in satiety and hydration during rest
        self.satiety -= 0.01 * rest_hours
        self.hydration -= 0.01 * rest_hours
        
        self._clamp_stats()
        return True
        
    def get_action_space(self) -> Dict[str, bool]:
        """Get possible actions based on current state
        
        Returns:
            Dict[str, bool]: Map of action names to availability
        """
        can_eat = any(item.item_type in [ItemType.FRUIT, ItemType.FISH, ItemType.MEAT] 
                     for item in self.inventory)
        can_drink = any(item.item_type == ItemType.WATER for item in self.inventory)
        can_heal = any(item.item_type == ItemType.MEDICINE 
                      and item.effects.get("health", 0) > 0 
                      for item in self.inventory)
        can_sleep = self.stress <= 0.8
        
        has_craft_materials = False
        material_count = 0
        for item in self.inventory:
            if item.item_type in [ItemType.STONE, ItemType.WOOD, ItemType.FIBER]:
                material_count += 1
                if material_count >= 2:
                    has_craft_materials = True
                    break
        
        # Check if any water/tent/item is visible
        water_visible = False
        tent_visible = False
        item_visible = False
        
        if self.environment:
            # Check for visible water tiles
            visible_tiles = self.environment.get_visible_tiles(self.environment.agent_pos)
            water_visible = any(tile == TileType.RIVER for tile in visible_tiles.values())
            tent_visible = any(tile == TileType.TENT for tile in visible_tiles.values())
            item_visible = len(self.environment.get_visible_items(self.environment.agent_pos)) > 0
        
        return {
            "eat": can_eat and self.satiety < 0.7,
            "drink": can_drink and self.hydration < 0.7,
            "heal": can_heal and self.health < 0.7,
            "sleep": can_sleep and (self.rest < 0.5 or self.stamina < 0.3),
            "explore": self.stamina > 0.3,
            "craft": self.stamina > 0.2 and len(self.inventory) > 1,
            # New reinforcement learning actions
            "move_to_water_and_drink": water_visible and self.hydration < 0.8 and self.stamina > 0.2,
            "collect_nearest_item": item_visible and len(self.inventory) < self.max_inventory_size and self.stamina > 0.2,
            "craft_tool": has_craft_materials and self.stamina > 0.2,
            "move_to_tent_and_rest": tent_visible and (self.rest < 0.6 or self.stamina < 0.4) and self.stamina > 0.1
        }
            
    def _check_item_index(self, index: int) -> bool:
        """Check if item index is valid"""
        return 0 <= index < len(self.inventory)
    
    def get_status_report(self) -> Dict[str, float]:
        """Get current status report
        
        Returns:
            Dict[str, float]: Current status values
        """
        return {
            "health": self.health,
            "satiety": self.satiety,
            "hydration": self.hydration,
            "temperature": self.temperature,
            "stamina": self.stamina,
            "rest": self.rest,
            "stress": self.stress,
        }
    
    def get_inventory_report(self) -> List[str]:
        """Get inventory report
        
        Returns:
            List[str]: List of inventory item descriptions
        """
        total_weight = sum(item.weight for item in self.inventory)
        report = [f"Inventory ({len(self.inventory)}/{self.max_inventory_size}, Weight: {total_weight:.1f}/{self.max_weight}kg):"]
        
        for i, item in enumerate(self.inventory):
            report.append(f"{i}: {item}")
            
        return report
    
    def perform_rl_action(self, action_name: str) -> Tuple[bool, float]:
        """Execute action for reinforcement learning
        
        Args:
            action_name: Name of action to execute
            
        Returns:
            Tuple[bool, float]: (success status, reward value)
        """
        # Check if action is available
        available_actions = self.get_action_space()
        if action_name not in available_actions or not available_actions[action_name]:
            return False, -0.1  # Negative reward for selecting unavailable action
        
        # Execute action and calculate reward
        if action_name == "eat":
            success = self.eat()
            # Reward based on satiety increase
            reward = 0.3 * success
            return success, reward
            
        elif action_name == "move_to_water_and_drink":
            prev_hydration = self.hydration
            success = self.move_to_water_and_drink()
            # Reward proportional to hydration increase
            hydration_gain = max(0, self.hydration - prev_hydration)
            reward = 0.5 * hydration_gain if success else -0.05
            return success, reward
            
        elif action_name == "collect_nearest_item":
            prev_inv_size = len(self.inventory)
            success = self.collect_nearest_item()
            # Reward for item acquisition
            if success and len(self.inventory) > prev_inv_size:
                reward = 0.4
            else:
                reward = -0.05
            return success, reward
            
        elif action_name == "move_to_tent_and_rest":
            prev_rest = self.rest
            prev_stamina = self.stamina
            success = self.move_to_tent_and_rest()
            # Reward proportional to rest and stamina recovery
            rest_gain = max(0, self.rest - prev_rest)
            stamina_gain = max(0, self.stamina - prev_stamina)
            reward = 0.3 * (rest_gain + stamina_gain) if success else -0.05
            return success, reward
            
        else:
            return False, 0.0  # Unknown action
    
    def get_state_for_learning(self) -> Dict[str, float]:
        """Get state vector for reinforcement learning
        
        Returns:
            Dict[str, float]: State vector for learning
        """
        # Get basic status
        state = self.get_status_report()
        
        # Add inventory information
        state['inventory_count'] = len(self.inventory)
        state['inventory_weight'] = sum(item.weight for item in self.inventory)
        state['inventory_fullness'] = len(self.inventory) / self.max_inventory_size
        
        # Add environment information
        if self.environment:
            # Check if water, tent, or items are visible
            visible_tiles = self.environment.get_visible_tiles(self.environment.agent_pos)
            state['water_visible'] = int(any(tile == TileType.RIVER for tile in visible_tiles.values()))
            state['tent_visible'] = int(any(tile == TileType.TENT for tile in visible_tiles.values()))
            state['items_visible'] = int(len(self.environment.get_visible_items(self.environment.agent_pos)) > 0)
            
            # Current tile information
            current_tile = self.environment.get_tile_at_position(*self.environment.agent_pos)
            for tile_type in [TileType.BEACH, TileType.GRASS, TileType.RIVER, TileType.MOUNTAIN]:
                state[f'on_{tile_type.name.lower()}'] = int(current_tile == tile_type)
        else:
            # Default values when environment is not available
            state['water_visible'] = 0
            state['tent_visible'] = 0
            state['items_visible'] = 0
            state['on_beach'] = 0
            state['on_grass'] = 0
            state['on_river'] = 0
            state['on_mountain'] = 0
            
        return state
    
    def __str__(self) -> str:
        """Return string representation of survivor status"""
        status = []
        status.append(f"=== {self.name} (Day {self.days_survived}) ===")
        status.append(f"Health: {self.health:.2f}")
        status.append(f"Satiety: {self.satiety:.2f}")
        status.append(f"Hydration: {self.hydration:.2f}")
        status.append(f"Temperature: {self.temperature:.2f}")
        status.append(f"Stamina: {self.stamina:.2f}")
        status.append(f"Rest: {self.rest:.2f}")
        status.append(f"Stress: {self.stress:.2f}")
        
        return "\n".join(status)
