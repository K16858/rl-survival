extends Node2D


var world_object_array:Array = []

const WORLD_MAX_X:int = 100;
const WORLD_MAX_Y:int = 100;

const X_per_Tile = 50;
const Y_pre_Tile = 50;

enum MapTileEnum {
	ground,
	glass,
	sand,
	river,
	ocean
}

@onready var MapTileNode:TileMapLayer = $MapTile

func _ready():
	for i in range(WORLD_MAX_X):
		var row = []
		for j in range(WORLD_MAX_Y):
			row.push_back(99)
		world_object_array.append(row)
	
	for i in range(WORLD_MAX_X):
		for l in range(WORLD_MAX_Y):
			var data = MapTileNode.get_cell_tile_data(Vector2(i-50,l-50))
			if data:
				world_object_array[i][l] = data.get_custom_data("kind")
			
		
	
