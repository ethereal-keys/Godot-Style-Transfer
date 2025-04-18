extends MyCustomNode

func _ready():
	set("viewport_path", NodePath("../Game"))
	set("display_sprite_path", NodePath("../Sprite2D"))
