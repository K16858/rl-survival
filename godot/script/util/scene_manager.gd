extends Node

var scene_dictionary:Dictionary[String,PackedScene] = {
	"main":preload("res://scene/main.tscn")
}

func change_scene(key:String):
	get_tree().change_scene_to_packed(scene_dictionary[key]);
