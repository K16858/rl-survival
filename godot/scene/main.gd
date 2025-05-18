extends Node2D


var world_tile_array: Array = []
var world_object_array:Array = []

var day: int = 0;
var time: float = 0;

var delta_count:float

var thread:Thread;

const EXE_PATH:String = "./server.exe"

const WORLD_MAX_X: int = 100;
const WORLD_MAX_Y: int = 100;

const X_per_Tile = 50;
const Y_pre_Tile = 50;

const ITEM_MARGIN:int = 4;

const SECONDperDAY:float = 60;

enum MapTileEnum {
	ground,
	glass,
	sand,
	river,
	ocean
}

@onready var MapTileNode: TileMapLayer = $MapTile
@onready var ItemTileNode: TileMapLayer = $ItemTile

func _ready():
	
	thread =Thread.new()
	
	thread.start(OS.execute.bind(EXE_PATH, []))
	
	day = 1;
	time = 0;
	for i in range(WORLD_MAX_X):
		var row = []
		for j in range(WORLD_MAX_Y):
			row.push_back(99)
		world_tile_array.append(row)
	
	for i in range(WORLD_MAX_X):
		var row = []
		for j in range(WORLD_MAX_Y):
			row.push_back(99)
		world_object_array.append(row)
	
	for i in range(WORLD_MAX_X):
		for l in range(WORLD_MAX_Y):
			var data = MapTileNode.get_cell_tile_data(Vector2(i - 50, l - 50))
			if data:
				world_tile_array[i][l] = data.get_custom_data("kind")
	
	var item_margin_counter:int = 0;
	for i in range(WORLD_MAX_X):
		for l in range(WORLD_MAX_Y):
			if(item_margin_counter <= 0 and world_tile_array[i][l] != 3 and world_tile_array[i][l] != 4):
				if (randf_range(0,100) >= 50):
					item_margin_counter = ITEM_MARGIN;
					if(randi_range(0,3) == 1):
						var item_kind = randi_range(6,7);
						change_item_layer(i,l,item_kind);
					else:
						var item_kind = randi_range(0,4);
						change_item_layer(i,l,item_kind);
			else:
				world_object_array[i][l] = 0;
			
			item_margin_counter -= 1;
			
func change_item_layer(x:int,y:int,kind:int):
	world_object_array[x][y] = kind
	ItemTileNode.set_cell(Vector2i(x-50,y-50),0,Vector2i(kind-1,0))
	
func delete_item_layer(x:int,y:int):
	world_object_array[x][y] = 0
	ItemTileNode.erase_cell(Vector2i(x-50,y-50))
	
func _process(delta):
	delta_count += delta;
	if(delta_count >= 1):
		delta_count -= 1;
		time +=1;
		
		if(time >= SECONDperDAY):
			time += 60;
			day += 1;
			$Player/Info/Day/Time.text = "Day:" + str(day);
