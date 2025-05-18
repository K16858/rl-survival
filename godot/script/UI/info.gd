class_name  Info

extends Node2D

@onready var PlayerNode:Player = $"../../Player"
@onready var DayTimeNode:Label = $"Day/Time"
@onready var PowerLabelNode:Label = $"Power/PowerLabel"

const POWER_MAX:int = 100;

var power:int;
var delta_count:float = 0


func _ready():
	power = 10;
	delta_count = 0;
	
	#脱出ボタンのシグナルを受け取る
	$"Menu/Panel/Escape".connect("pressed",_escape_pressed)

func _process(delta):
	delta_count += delta;
	PowerLabelNode.text = str(power) + "/" + str(POWER_MAX)
	
	if(delta_count >= 1):
		delta_count -= 1;
		power = min(power+1,POWER_MAX)


func _on_show_status_pressed():
	visible = !visible
	
func _escape_pressed():
	if(power >= 100):
		print("YouWin!!!!!!!!!!!!!!")
		$InfoSE.play_se("recover")
		SceneManager.change_scene("game_clear")
	else:
		$InfoSE.play_se("failure")


func _on_player_gameover():
	visible = false;
