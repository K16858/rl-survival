class_name Resource_t

var _resource: float = 1

func _init():
	_resource = 1

func getres() -> float:
	return _resource

func addres(a: float):
	_resource = min(_resource + a, 1)

# リソースが0になるとtrueを返す
func subres(s: float)-> bool:
	_resource = max(_resource - s, 0)
	return _resource == 0.0
