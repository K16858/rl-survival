class_name  StatusIcons

extends Node2D

@export var kind:int;

@onready var IconList:Array[AnimatedSprite2D] = [
	$"Icon1",
	$"Icon2",
	$"Icon3",
	$"Icon4",
	$"Icon5"
]


func _ready():
	for i in IconList:
		i.frame = kind
		


func update_icon(phase:int):
	for i in range(IconList.size()):
		IconList[i].visible = false;
	if(phase >= 0):
		for i in range(phase):
			IconList[i].visible = true;
