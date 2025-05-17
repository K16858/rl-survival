class_name Player

extends Node2D

@export var speed: float = 200.0
@export var server_url: String = "http://127.0.0.1:8000/reasoning"
@export var map: TileMapLayer

signal  update_status(status:Array[float])

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

var delta_count:float

func _on_request_completed(result, response_code, headers, body):
	var json = JSON.parse_string(body.get_string_from_utf8())
	print(json)
	if (json["next"]["kind"] == "move"):
		_move(json["next"]["x"], json["next"]["y"])
	#_send_request()

func _ready():
	# http要求関連の初期化
	$HTTPRequest.request_completed.connect(_on_request_completed)
	
	# プロパティ初期化
	hp = Resource_t.new()
	hp._init()
	satiety = Resource_t.new()
	satiety._init()
	nthirsty = Resource_t.new()
	nthirsty._init()
	body_temperature = Resource_t.new()
	body_temperature._init()
	stamina = Resource_t.new()
	stamina._init()
	drowsiness = Resource_t.new()
	drowsiness._init()
	stress = Resource_t.new()
	stress._init()
	# その他初期化
	$AnimatedSprite2D.play("default")
	
	# 初期行動要求
	_send_request()
	
	delta_count = 0;

func _process(delta):
	#一秒毎にステータスのUIを同期
	delta_count += delta;
	if(delta_count > 1):
		delta_count -= 1;
		update_status.emit([hp.getres(),satiety.getres(),nthirsty.getres(),body_temperature.getres(),stamina.getres(),drowsiness.getres(),stress.getres()]);
	

# タイルマップ上の移動操作
# rel = falseで絶対座標で行動
func _move(x: int, y: int, rel: bool = true):
	if rel:
		var distination_pos = map.map_to_local(Vector2(x, y))
		var distance = distination_pos.distance_to(position)
		var tween = self.create_tween()
		tween.tween_property(self, "position", distination_pos, distance / speed)
		tween.tween_callback(_send_request)
	else:
		print("not yet impled non rel move")


func _send_request():
	const header = ["Content-Type: application/json"]
	var data = {
		"hp": hp.getres(),
		"satiety": satiety.getres(),
		"nthirsty": nthirsty.getres(),
		"body_temperature": body_temperature.getres(),
		"stamina": stamina.getres(),
		"drowsiness": drowsiness.getres(),
		"stress": stress.getres()
	}
	var error = $HTTPRequest.request(server_url, header, HTTPClient.METHOD_POST, JSON.stringify(data))
	if error != OK:
		push_error("An error occurred in the HTTP request.")
	print("send")
