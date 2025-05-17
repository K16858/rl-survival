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
		
