extends Node2D

var world_object_array

const WORLD_MAX_X:int = 100;
const WORLD_MAX_Y:int = 100;

func _ready():
	world_object_array.resize(100)
	for i in world_object_array:
		i.resize(100)
