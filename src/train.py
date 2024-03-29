import argparse
import src.trainer_main as trainer_main
parser = argparse.ArgumentParser(description='Whats cooking')
parser.add_argument('--train_file',default='/home/naraya01/AEN/GIT/Whats_cooking/Dataset/train.json',type=str,metavar='N',help='Path to config file.')
parser.add_argument('--test_file',default='/home/naraya01/AEN/GIT/Whats_cooking/Dataset/test.json',type=str,metavar='N',help='Path to config file.')
parser.add_argument('--method',default='MLP',type=str,metavar='N',help='Path to config file.')
parser.add_argument('--export_folder',default='/home/naraya01/AEN/GIT/Whats_cooking/Model/')
parser.add_argument('--mode',default='train')
parser.add_argument('--model',default='/home/naraya01/AEN/GIT/Whats_cooking/Model/2019-07-23 09:33:16/RF.joblib')

if __name__ == '__main__':
    args = parser.parse_args()
    trainer = trainer_main.trainer_main(args)

