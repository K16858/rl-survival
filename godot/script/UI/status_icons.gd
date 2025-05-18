class_name  StatusIcons

extends Node2D

@export var kind:int
@export var COST:int = 10;

@onready var IconList:Array[AnimatedSprite2D] = [
	$"Icon1",
	$"Icon2",
	$"Icon3",
	$"Icon4",
	$"Icon5"
]

@onready var InfoNode:Info = $"../../../Info"
@onready var StatusNode:Status = $"../../Status"


func _ready():
	for i in IconList:
		i.frame = kind
		


func update_icon(phase:int):
	for i in range(IconList.size()):
		IconList[i].visible = false;
	if(phase >= 0):
		for i in range(phase):
			IconList[i].visible = true;


func _on_clickable_area_mouse_entered():
	pass
	
func _on_clickable_area_mouse_exited():
	pass # Replace with function body.


func _on_clickable_area_input_event(viewport, event, shape_idx):
	if Input.is_action_just_released("click"):
		if(InfoNode.power >= COST):
			$"../../InfoSE".play_se("recover")
			InfoNode.power -= COST
			StatusNode.send_god_present.emit(kind)
		else:
			$"../../InfoSE".play_se("failure")
