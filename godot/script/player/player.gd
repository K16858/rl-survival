class_name Player

extends Node2D

@export var speed: float = 200.0
@export var server_url: String = "http://127.0.0.1:8000/reasoning"

# 体力
var hp: Resource_t
# 満腹度
var satiety: Resource_t
# のどの乾き具合の逆
var nthirsty: Resource_t
# 体温
var body_temperature: Resource_t
# スタミナ
var stamina: Resource_t
# 眠気
var drowsiness: Resource_t
# ストレス
var stress: Resource_t

func _on_request_completed(result, response_code, headers, body):
	var json = JSON.parse_string(body.get_string_from_utf8())
	print(json)
	if (json["next"]["kind"] == "move"):
		_move(json["next"]["x"], json["next"]["y"])
	_send_request()

func _ready():
	# http要求関連の初期化
	$HTTPRequest.request_completed.connect(_on_request_completed)
	
	# プロパティ初期化
	#hp = Resource_t()
	#satiety._init(1.)
	#nthirsty._init(1.)
	#body_temperature._init(1.)
	#stamina._init(1.)
	#drowsiness._init(0.)
	#stress._init(0.)
	# その他初期化
	$AnimatedSprite2D.play("default")
	
	# 初期行動要求
	_send_request()

func _process(delta):
	pass

func _move(x: int, y: int):
	position = Vector2(x*10, y*10)
	


func _send_request():
	const header = ["Content-Type: application/json"]
	var data = {
		"hp": hp,
		"satiety": satiety,
		"nthirsty": nthirsty,
		"body_temperature": body_temperature,
		"stamina": stamina,
		"drowsiness": drowsiness,
		"stress": stress
	}
	var error = $HTTPRequest.request(server_url, header, HTTPClient.METHOD_POST, JSON.stringify(data))
	if error != OK:
		push_error("An error occurred in the HTTP request.")
	print("send")
