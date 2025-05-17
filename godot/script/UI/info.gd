extends Node2D

@onready var PlayerNode:Player = $"../../Player"

func _ready():
	pass



func _on_show_status_pressed():
	visible = !visible
