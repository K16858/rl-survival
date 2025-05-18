from enum import IntEnum
from typing import Dict, List, Any, Optional, Tuple
import random

class TileType(IntEnum):
    """Type of terrain tiles"""
    OCEAN = 0  # Ocean (impassable)
    BEACH = 1  # Beach
    GRASS = 2  # Grassland
    RIVER = 3  # River
    FOREST = 4  # Forest
    MOUNTAIN = 5  # Mountain
    CAVE = 6  # Cave

    @staticmethod
    def get_color(tile_type):
        """Get the color for rendering a tile type"""
        colors = {
            TileType.OCEAN: (0, 105, 148),      # Deep blue
            TileType.BEACH: (194, 178, 128),    # Sand color
            TileType.GRASS: (34, 139, 34),      # Green
            TileType.RIVER: (64, 164, 223),     # Light blue
            TileType.FOREST: (0, 100, 0),       # Dark green
            TileType.MOUNTAIN: (139, 137, 137), # Gray
            TileType.CAVE: (65, 65, 65),        # Dark gray
        }
        return colors.get(tile_type, (100, 100, 100))  # Default gray
    
    @staticmethod
    def is_passable(tile_type) -> bool:
        """Check if a tile type can be traversed"""
        return tile_type != TileType.OCEAN

class ObjectType(IntEnum):
    """Type of objects that can be placed on tiles"""
    NONE = 0      # No object
    TREE = 1      # Tree (can be harvested for wood)
    BUSH = 2      # Bush (can contain berries)
    ROCK = 3      # Rock (can be harvested for stone)
    TENT = 4      # Tent (player-built shelter)
    CAMPFIRE = 5  # Campfire (for cooking and warmth)
    ITEM = 6      # Collectable item
    ANIMAL = 7    # Wildlife (can be hunted)
    FISH = 8      # Fish (in water)

class WorldObject:
    """Base class for objects that can be placed on tiles"""
    
    def __init__(self, 
                 obj_type: ObjectType,
                 name: str,
                 position: Tuple[int, int],
                 collectible: bool = False,
                 durability: float = 1.0,
                 interaction_text: str = "Examine"):
        self.obj_type = obj_type
        self.name = name
        self.position = position
        self.collectible = collectible  # Can be picked up
        self.durability = durability    # How many uses before destruction
        self.interaction_text = interaction_text
        
    def interact(self, agent=None) -> Dict[str, Any]:
        """Default interaction behavior
        
        Args:
            agent: The agent interacting with this object
            
        Returns:
            Dict containing interaction results
        """
        return {"message": f"You see a {self.name}.", "success": True}
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get color for rendering this object"""
        colors = {
            ObjectType.NONE: (0, 0, 0),          # Nothing/transparent
            ObjectType.TREE: (0, 80, 0),         # Darker green
            ObjectType.BUSH: (0, 150, 0),        # Lighter green
            ObjectType.ROCK: (120, 120, 120),    # Gray
            ObjectType.TENT: (150, 100, 150),    # Purple tent
            ObjectType.CAMPFIRE: (255, 140, 0),  # Orange fire
            ObjectType.ITEM: (255, 255, 0),      # Yellow for items
            ObjectType.ANIMAL: (150, 75, 0),     # Brown for animals
            ObjectType.FISH: (0, 190, 190),      # Cyan for fish
        }
        return colors.get(self.obj_type, (255, 255, 255))
    
    def __str__(self) -> str:
        return f"{self.name} at {self.position}"

class Tree(WorldObject):
    """Tree object that can be harvested for wood"""
    
    def __init__(self, position: Tuple[int, int], size: float = 1.0):
        super().__init__(
            ObjectType.TREE,
            f"{'Small' if size < 0.5 else 'Large'} Tree",
            position,
            collectible=False,
            durability=3.0 * size,  # Larger trees take more hits
            interaction_text="Chop"
        )
        self.size = size  # Size affects resources gained
        self.resources = {
            "wood": int(2 + size * 3),  # 2-5 wood based on size
            "leaf": int(1 + size * 2),  # 1-3 leaves based on size
        }
        
    def interact(self, agent=None) -> Dict[str, Any]:
        """Chop the tree for wood"""
        if self.durability <= 0:
            return {"message": "This tree has already been chopped down.", "success": False}
        
        # Reduce durability
        self.durability -= 1.0
        
        # Calculate resources to give
        wood_amount = min(2, self.resources["wood"])
        leaf_amount = min(1, self.resources["leaf"])
        
        # Reduce remaining resources
        self.resources["wood"] -= wood_amount
        self.resources["leaf"] -= leaf_amount
        
        # If agent provided, give them the resources
        collected = {}
        if agent and hasattr(agent, 'add_item'):
            # Later we'll handle actual item creation and inventory
            collected = {
                "wood": wood_amount,
                "leaf": leaf_amount
            }
            
        # Check if completely harvested
        destroyed = self.durability <= 0
            
        return {
            "message": f"You chopped the tree and got {wood_amount} wood and {leaf_amount} leaves.",
            "success": True,
            "collected": collected,
            "destroyed": destroyed
        }

class Bush(WorldObject):
    """Bush that can contain berries"""
    
    def __init__(self, position: Tuple[int, int]):
        super().__init__(
            ObjectType.BUSH,
            "Berry Bush",
            position,
            collectible=False,
            durability=2.0,
            interaction_text="Gather"
        )
        self.has_berries = random.random() < 0.7  # 70% chance to have berries
        
    def interact(self, agent=None) -> Dict[str, Any]:
        """Gather berries from the bush"""
        if not self.has_berries:
            return {"message": "This bush has no berries.", "success": False}
        
        # How many berries to collect
        berry_count = random.randint(1, 3)
        self.has_berries = False
        
        # If agent provided, give them berries
        collected = {}
        if agent and hasattr(agent, 'add_item'):
            collected = {"berries": berry_count}
            
        return {
            "message": f"You gathered {berry_count} berries from the bush.",
            "success": True,
            "collected": collected,
            "destroyed": False
        }

class Campfire(WorldObject):
    """Campfire for cooking and warmth"""
    
    def __init__(self, position: Tuple[int, int], lit: bool = True):
        super().__init__(
            ObjectType.CAMPFIRE,
            "Campfire",
            position,
            collectible=False,
            durability=10.0,  # Burns out after some time
            interaction_text="Use"
        )
        self.lit = lit
        self.heat_radius = 3  # Tiles affected by heat
        self.cook_enabled = lit
        
    def interact(self, agent=None) -> Dict[str, Any]:
        """Use the campfire"""
        if not self.lit:
            return {
                "message": "The campfire is not lit. You need to light it first.",
                "success": False,
                "actions": ["light"]
            }
            
        # Decrease durability as fire burns
        self.durability -= 0.5
        
        # Check if fire burned out
        if self.durability <= 0:
            self.lit = False
            self.cook_enabled = False
            return {
                "message": "The fire has burned out.",
                "success": True,
                "effects": {"warmth": 0},
                "destroyed": True
            }
        
        # Warmth effect for the agent
        effects = {"warmth": 0.3}
        
        # Apply effects to agent if provided
        if agent and hasattr(agent, 'temperature'):
            # Example: move temperature toward optimal range
            if agent.temperature < 0.5:
                agent.temperature = min(0.5, agent.temperature + 0.1)
        
        return {
            "message": "You warm yourself by the fire.",
            "success": True,
            "effects": effects,
            "actions": ["cook", "add_fuel"]
        }
    
    def add_fuel(self, amount: float = 1.0) -> Dict[str, Any]:
        """Add fuel to extend the campfire's life"""
        self.durability = min(10.0, self.durability + amount * 2)
        
        if not self.lit:
            self.lit = True
            self.cook_enabled = True
            return {"message": "You added fuel and lit the fire.", "success": True}
        
        return {"message": f"You added fuel to the fire.", "success": True}

class ItemObject(WorldObject):
    """Collectible item in the world"""
    
    def __init__(self, 
                 name: str, 
                 position: Tuple[int, int],
                 item_properties: Dict[str, Any] = None):
        super().__init__(
            ObjectType.ITEM,
            name,
            position,
            collectible=True,
            interaction_text="Pick up"
        )
        self.item_properties = item_properties or {}
        
    def interact(self, agent=None) -> Dict[str, Any]:
        """Pick up the item"""
        if agent and hasattr(agent, 'add_item'):
            # This would create an actual Item instance and add to inventory
            success = True  # Placeholder for actual inventory logic
            
            if success:
                return {
                    "message": f"You picked up the {self.name}.",
                    "success": True,
                    "collected": {self.name: 1},
                    "destroyed": True  # Item is removed from world when collected
                }
            else:
                return {
                    "message": f"You couldn't pick up the {self.name}. Your inventory might be full.",
                    "success": False
                }
        
        return {"message": f"You see a {self.name}.", "success": False}


class Tile:
    """Represents a single tile in the game world"""
    
    def __init__(self, 
                 tile_type: TileType, 
                 position: Tuple[int, int],
                 elevation: float = 0.0):
        self.tile_type = tile_type
        self.position = position
        self.elevation = elevation  # Height of this tile (for terrain generation)
        self.objects: List[WorldObject] = []  # Objects on this tile
    
    def add_object(self, obj: WorldObject) -> bool:
        """Add an object to this tile"""
        # Check if we can add more objects (simple rule: max 3 objects per tile)
        if len(self.objects) >= 3:
            return False
            
        # Update object position to match tile
        obj.position = self.position
        self.objects.append(obj)
        return True
    
    def remove_object(self, obj: WorldObject) -> bool:
        """Remove an object from this tile"""
        if obj in self.objects:
            self.objects.remove(obj)
            return True
        return False
    
    def get_objects_by_type(self, obj_type: ObjectType) -> List[WorldObject]:
        """Get all objects of a specific type on this tile"""
        return [obj for obj in self.objects if obj.obj_type == obj_type]
        
    def is_passable(self) -> bool:
        """Check if this tile can be traversed"""
        # First check the base tile type
        if not TileType.is_passable(self.tile_type):
            return False
            
        # Then check if any objects block passage (like large rocks)
        for obj in self.objects:
            # Example rule: trees and rocks block passage
            if obj.obj_type in [ObjectType.TREE, ObjectType.ROCK] and obj.durability > 0:
                return False
                
        return True
    
    def get_description(self) -> str:
        """Get a description of this tile and its contents"""
        tile_names = {
            TileType.OCEAN: "ocean",
            TileType.BEACH: "sandy beach",
            TileType.GRASS: "grassy field",
            TileType.RIVER: "flowing river",
            TileType.FOREST: "dense forest",
            TileType.MOUNTAIN: "rocky mountain",
            TileType.CAVE: "dark cave"
        }
        
        desc = f"You are at a {tile_names.get(self.tile_type, 'unknown terrain')}."
        
        if self.objects:
            obj_desc = []
            for obj in self.objects:
                obj_desc.append(f"a {obj.name}")
            desc += f" There is {', '.join(obj_desc)} here."
        
        return desc
        
    def get_color(self) -> Tuple[int, int, int]:
        """Get color for rendering this tile"""
        return TileType.get_color(self.tile_type)
