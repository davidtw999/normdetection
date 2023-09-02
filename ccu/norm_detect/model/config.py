import torch, os

CLASS_MAP_NORM = {
        "101":'apology',
        "102":'criticism',
        "103":'greeting',
        "104":"request",
        "105": "persuasion",
        "106": "thanking",
        "107": "farewell",
        "noann": "noann",
        'none': 'none'
    }

LABEL2ID_NORM = {'101': 0, '102': 1, '103': 2, '104':3, '105':4, '106':5, '107':6, 'none':7, "noann": 8}
ID2LABEL_NORM = {v:k for k,v in LABEL2ID_NORM.items()}
LABEL2ID_STATUS = {'adhere': 0, 'violate':1, 'noann': 2, 'EMPTY_NA': 3}
ID2LABEL_STATUS = {v:k for k,v in LABEL2ID_STATUS.items()}

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_train_epochs = 20
    train_file_path = "../data/norm/train.tab"
    dev_file_path = "../data/norm/dev.tab"
    test_file_path = "../data/norm/test.tab"

    model_path = './roberta-zh-sensible'
    train_batch_size = 16
    eval_batch_size = 128
    num_labels = 9
    wandboff = True
    project_name = 'CCU'

    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    seed = 2019

    filename = model_path.split('/')[-1] + '_lr' + str(learning_rate) + '_bs' + str(train_batch_size) + '_epoch' + str(
        num_train_epochs) + '_seed' + str(seed)

    model_use = 'test'
    ckpt_dir = './save_model'

    continue_train = False
    continue_num_train_epochs = 20
    saved_model_file = os.path.join(ckpt_dir, filename) + '_ep1'