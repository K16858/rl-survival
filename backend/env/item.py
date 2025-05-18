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
    TENT = 11       # Tent

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

class Tent(Item):
    """Tent item that can be deployed for shelter and rest"""
    
    def __init__(self, name: str = "Camping Tent", weight: float = 2.5, durability: float = 5.0):
        effects = {
            "energy": 0.3,  # Energy recovery when used
            "shelter": 0.8,  # Protection from weather
            "comfort": 0.7   # Comfort level for recovery
        }
        
        super().__init__(
            item_type=ItemType.TENT,
            name=name,
            weight=weight,
            durability=durability,
            effects=effects
        )
        self.deployed = False  # Tracks if tent is currently set up
        self.capacity = 1      # Number of people it can shelter
    
    def use(self) -> Dict[str, float]:
        """Deploy tent or rest in it if already deployed"""
        if self.durability <= 0:
            return {}  # Broken tent has no effect
        
        if not self.deployed:
            self.deployed = True
            return {"message": "Tent has been set up. You can now rest in it."}
        else:
            # Small durability reduction when used
            self.durability -= 0.1
            if self.durability < 0:
                self.durability = 0.0
                
            return self.effects
    
    def pack(self) -> bool:
        """Pack up the tent for travel"""
        if not self.deployed:
            return False
            
        self.deployed = False
        return True
    
    def repair(self, repair_amount: float = 0.2) -> bool:
        """Repair tent with materials"""
        if self.durability >= 1.0:
            return False  # Already at max durability
            
        self.durability = min(1.0, self.durability + repair_amount)
        return True
    
    def __str__(self) -> str:
        status = "Deployed" if self.deployed else "Packed"
        return f"{self.name} ({status}, Durability: {self.durability:.1f})"
    