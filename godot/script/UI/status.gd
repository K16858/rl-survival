extends Node2D

@onready var IconList:Array[StatusIcons] = [
	$"HP",
	$"Satiety",
	$"Nthirsty",
	$"BodyTemperature",
	$"Stamina",
	$"Drowsiness",
	$"Stress"
]

func _ready():
	pass
	


func _on_player_update_status(status):
	for i in range(status.size()):
		var temp:float = status[i];
		var count:int = -1; 
		while(temp>0):
			count +=1;
			temp -= 0.2;
		
		IconList[i].update_icon(count);
		
