import atexit
import shutil

from tensorboardX import SummaryWriter

writer: SummaryWriter = None
e: int = None
train_iter : int = 0
test_iter : int = 0

@atexit.register
def del_tensorboard():
    option = input("Delete?")
    if option.lower() == 'd':
        shutil.rmtree(writer.log_dir)
        print("DELETED!")
