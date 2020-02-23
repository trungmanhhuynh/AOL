from datetime import datetime
import sys

# abstraction for logging
class Logger():
    def __init__(self, args , mode = "train"):
        
        self.mode = mode
        if(self.mode == "train"):
            # open file for recording train loss 
            self.train_log_fn = '{}/train_log_{}.txt'.format(args.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 
            self.train_log = open(self.train_log_fn, 'w')
        elif(self.mode == "test"): 
            self.test_log_fn = '{}/test_log_{}.txt'.format(args.log_dir,datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 
            self.test_log = open(self.test_log_fn, 'w')
        elif(self.mode == "test_cn"): 
            self.test_cn_log_fn = '{}/test_cn_log_{}.txt'.format(args.log_dir,datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 
            self.test_cn_log = open(self.test_cn_log_fn, 'w')
        else: 
            sys.exit("wrong logger mode")


    def write(self, s, record_loss = False):

        print (s)
        if(self.mode == "train"):
            with open(self.train_log_fn, 'a') as f:
                f.write(s + "\n")
        elif(self.mode == "test"): 
            with open(self.test_log_fn, 'a') as f:
                f.write(s + "\n")
        elif(self.mode == "test_cn"): 
            with open(self.test_cn_log_fn, 'a') as f:
                f.write(s + "\n")
        else:            
            sys.exit("wrong logger mode")
