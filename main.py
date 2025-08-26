import luigi
from src.tasks.train import TrainTask

# Luigi UIに表示されるようにWorkerモードで実行
if __name__ == "__main__":
    settings_file_path = "configs/settings.yml"

    # Luigi UIで確認可能な方法で実行
    luigi.build(
        [TrainTask(settings_file_path=settings_file_path)],
        scheduler_host="localhost",
        scheduler_port=8082,
        workers=1,
    )
