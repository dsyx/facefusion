from custom.patch import content_analyser, environment


def setup() -> None:
	environment.setup()
	content_analyser.setup()
