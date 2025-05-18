extends AudioStreamPlayer

var SEList:Dictionary[String,AudioStreamMP3] = {
	"eat":preload("res://assets/SE/eat.mp3"),
	"drink":preload("res://assets/SE/drink.mp3")
}

func play_se(key:String):
	stream = SEList[key]
	play()
