from custom.patch import content_analyser, environment, inference_manager


def setup() -> None:
	environment.setup()
	content_analyser.setup()
	inference_manager.setup()
