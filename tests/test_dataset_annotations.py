from dataset.caption import auto_captioner


def test_caption_annotations_keep_action_environment_and_conditions():
    caption = auto_captioner.generate_caption("A red car driving on a highway during heavy rain at night, tracking shot")
    assert "driving" in caption.actions
    assert caption.environment == "highway"
    assert caption.weather == "rainy overcast"
    assert caption.time == "night"
    assert caption.camera == "tracking shot"
