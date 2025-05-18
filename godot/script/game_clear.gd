extends Node2D

func _ready():
	$AnimationPlayer.play("win")
	$Win.visible = false
	$Button.visible = false
	

func anim_finished():
	$Win.visible = true
	$Button.visible = true

func _on_button_pressed():
	pass # Replace with function body.
