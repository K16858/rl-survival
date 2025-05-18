class_name Resource_t

var _resource: float

func _init():
	pass
	
static func newres(i: float = 0) -> Resource_t:
	var instance = Resource_t.new()
	instance._resource = i
	return instance

func getres() -> float:
	return _resource

func addres(a: float):
	_resource = clampf(_resource + a, 0, 1)

# リソースが0になるとtrueを返す
func subres(s: float)-> bool:
	_resource = clampf(_resource - s, 0, 1)
	return _resource == 0.0
