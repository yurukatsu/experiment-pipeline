import logging

import gokart

from src.tasks.train import TrainTask


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# Test with DummyRegressor
settings_file_path = "configs/settings.yml"
dummy_task = TrainTask(settings_file_path=settings_file_path)

print("Testing with DummyRegressor...")
gokart.build(dummy_task, return_value=False)
