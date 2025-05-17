from enum import Enum
from typing import Dict, List, Any, Tuple
import numpy as np
import random

class ItemType(Enum):
    """Enum defining item types"""
    NONE = 0
    WOOD = 1        # Wood
    STONE = 2       # Stone
    LEAF = 3        # Leaf
    FRUIT = 4       # Fruit
    FISH = 5        # Fish
    WATER = 6       # Water
    MEAT = 7        # Meat
    TOOL = 8        # Tool
    WEAPON = 9      # Weapon
    MEDICINE = 10   # Medicine

class Item:
    """Item class"""
    def __init__(self, 
                 item_type: ItemType, 
                 name: str, 
                 weight: float = 1.0, 
                 durability: float = 1.0,
                 effects: Dict[str, float] = None):
        self.item_type = item_type
        self.name = name
        self.weight = weight  # Weight (kg)
        self.durability = durability  # Durability (0-1)
        # Effects when item is used {stat_name: change_value}
        self.effects = effects if effects is not None else {}
    
    def use(self) -> Dict[str, float]:
        """Use the item and return effects"""
        if self.durability <= 0:
            return {}  # Broken items have no effect
            
        # Reduce durability for consumable items
        if self.item_type in [ItemType.FRUIT, ItemType.FISH, ItemType.WATER, ItemType.MEAT, ItemType.MEDICINE]:
            self.durability = 0.0  # Consumed entirely
            
        # Tools and weapons degrade slightly with use
        elif self.item_type in [ItemType.TOOL, ItemType.WEAPON]:
            self.durability -= 0.1
            if self.durability < 0:
                self.durability = 0.0
                
        return self.effects
    
    def __str__(self) -> str:
        return f"{self.name} (Durability: {self.durability:.1f})"

