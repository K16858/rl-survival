extends Node2D

var is_opend:bool = false;

const CLOSE_POSITION:Vector2 = Vector2(454.0,176.0)
const OPEN_POSITION:Vector2 = Vector2(-144,176)

func  _ready():
	is_opend = false;
	position = CLOSE_POSITION;


func _on_open_button_pressed():
	if(!is_opend):
		var tween = self.create_tween()
		tween.tween_property(self, "position", OPEN_POSITION, 0.2)
		is_opend = true
	else:
		var tween = self.create_tween()
		tween.tween_property(self, "position", CLOSE_POSITION, 0.2)
		is_opend = false


func _on_escape_pressed():
	SceneManager.change_scene("game_clear")
