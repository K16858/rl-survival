class_name Player

extends Node2D

@export var speed: float = 200.0
@export var server_url: String = "http://127.0.0.1:8000/reasoning"
@export var map: TileMapLayer

@export var resource_baseline: float = 0.7

@export var hp_change_multi: float = 0.3

@export var satiety_use_cycle: int = 50
@export var nthirsty_use_cycle: int = 30
@export var ndrowsiness_use_cycle: int = 100


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
# 眠気の逆
var ndrowsiness: Resource_t
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
	hp = Resource_t.newres(1)
	satiety = Resource_t.newres(1)
	nthirsty = Resource_t.newres(1)
	body_temperature = Resource_t.newres(1)
	stamina = Resource_t.newres(1)
	ndrowsiness = Resource_t.newres(1)
	stress = Resource_t.newres(0)
	# その他初期化
	$AnimatedSprite2D.play("default")
	
	# 初期行動要求
	_send_request()
	
	delta_count = 0;

func _process(delta):
	#一秒毎にステータスのUIを動機
	delta_count += delta;
	if(delta_count > 1):
		delta_count -= 1;
		update_status.emit([hp.getres(),satiety.getres(),nthirsty.getres(),body_temperature.getres(),stamina.getres(),ndrowsiness.getres(),stress.getres()]);
		
		# 各ステータスの変動
		# 時間経過ステータス
		satiety.subres(1./satiety_use_cycle)
		nthirsty.subres(1./nthirsty_use_cycle)
		ndrowsiness.subres(1./ndrowsiness_use_cycle)
		
		# 気温処理
		# TODO
		
		# 体力変動fnc
		var sub = func(x: float):
			return x - resource_baseline
		var stat_min = [satiety.getres(), nthirsty.getres(), ndrowsiness.getres(), (1-stress.getres())].map(sub).min()
		print(stat_min)
		
		if hp.subres(-stat_min * hp_change_multi):
			print("game over !!!!!!!!!!!!!!!!!")
	

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
		"ndrowsiness": ndrowsiness.getres(),
		"stress": stress.getres()
	}
	var error = $HTTPRequest.request(server_url, header, HTTPClient.METHOD_POST, JSON.stringify(data))
	if error != OK:
		push_error("An error occurred in the HTTP request.")
	print("send")
