extends AudioStreamPlayer

var SEList:Dictionary[String,AudioStreamMP3] = {
	"recover":preload("res://assets/SE/recover.mp3"),
	"failure":preload("res://assets/SE/failure.mp3"),
}

func play_se(key:String):
	stream = SEList[key]
	play()
