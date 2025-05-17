class_name Player

extends Node2D

@export var speed: float = 200.0
@export var server_url: String = "http://127.0.0.1:8000/"

func _on_request_completed(result, response_code, headers, body):
	var json = JSON.parse_string(body.get_string_from_utf8())
	print(json)
	if (json["next"]["kind"] == "move"):
		_move(json["next"]["x"], json["next"]["y"])
	_send_request()

func _ready():
	$HTTPRequest.request_completed.connect(_on_request_completed)
	$AnimatedSprite2D.play("default")
	_send_request()
	
func _process(delta):
	pass

func _move(x: int, y: int):
	position = Vector2(x*10, y*10)
	


func _send_request():
	var error = $HTTPRequest.request(server_url)
	if error != OK:
		push_error("An error occurred in the HTTP request.")
	print("send")
