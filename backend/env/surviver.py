from enum import Enum
from typing import Dict, List, Any, Tuple
import numpy as np
import random
from item import Item, ItemType

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
        # Improve crafting skill
        self.skills["crafting"] = min(1.0, self.skills["crafting"] + 0.01)
        
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
        
        return {
            "eat": can_eat and self.satiety < 0.7,
            "drink": can_drink and self.hydration < 0.7,
            "heal": can_heal and self.health < 0.7,
            "sleep": can_sleep and (self.rest < 0.5 or self.stamina < 0.3),
            "explore": self.stamina > 0.3,
            "craft": self.stamina > 0.2 and len(self.inventory) > 1
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


# Example usage
if __name__ == "__main__":
    # Sample item definitions
    apple = Item(
        ItemType.FRUIT, "Apple", 0.2, 1.0, 
        {"health": 0.05, "satiety": 0.2, "hydration": 0.1}
    )
    
    water_bottle = Item(
        ItemType.WATER, "Water Bottle", 0.5, 1.0,
        {"hydration": 0.5}
    )
    
    stone_axe = Item(
        ItemType.TOOL, "Stone Axe", 2.0, 1.0,
        {}
    )
    
    # Create and test survivor
    player = Survivor("Test Player")
    print(player)
    
    # Pick up items
    player.add_item(apple)
    player.add_item(water_bottle)
    player.add_item(stone_axe)
    print("\nAfter picking up items:")
    print("\n".join(player.get_inventory_report()))
    
    # Simulate time passing
    print("\nAfter 3 hours:")
    player.update(3.0)
    print(player)
    
    # Eat food
    print("\nAfter eating apple:")
    player.eat(0)  # Use apple
    print(player)
    print("\n".join(player.get_inventory_report()))
    
    # Drink water
    print("\nAfter drinking water:")
    player.drink(0)  # Use water (apple was consumed so water is now at index 0)
    print(player)
    print("\n".join(player.get_inventory_report()))
