extends Node2D

func _ready():
	visible = false

func _on_player_gameover():
	await get_tree().create_timer(1).timeout
	visible = true;
	$Button.visible = false;
	$AnimationPlayer.play("gameover")


func _on_animation_player_animation_finished(anim_name):
	if(anim_name == "gameover"):
		$Button.visible = true


func _on_button_pressed():
	pass # Replace with function body.
