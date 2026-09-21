import pytest

from app.services.prompt_service import prompt_service
from app.services.prompt_constraints import prompt_constraint_service
from app.models.scene import Scene


@pytest.mark.asyncio
async def test_red_car_constraints_survive_prompt_analysis():
    analysis = await prompt_service.understand_prompt(
        "Create a realistic video of a red sports car driving on a highway during heavy rain at night."
    )
    constraints = analysis["prompt_constraints"]
    assert constraints["subject"] == "red sports car"
    assert constraints["action"] == "driving"
    assert constraints["environment"] == "highway"
    assert constraints["weather"] == "heavy rain"
    assert constraints["time"] == "night"


def test_constraints_are_injected_into_every_scene_prompt():
    constraints = prompt_constraint_service.build("A red car driving on a highway during heavy rain at night")
    scene = Scene(1, "car", "narration", "basic scene", 2.0, metadata={"enriched_prompt": "base"})
    prompt_constraint_service.apply_to_scenes([scene], constraints)
    assert "red car" in scene.metadata["enriched_prompt"].lower()
    assert "dry road" in scene.metadata["negative_prompt"]
