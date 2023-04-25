import numpy as np


class ReturnStats:
    """
    This class store objective and loss on the train an test set.
    In particular, for all of them, the class have two values destined to be
    the mean and standard deviation if there is more than one value.
    """
    def __init__(self, 
                 train_obj = None, 
                 test_obj = None, 
                 train_loss = None, 
                 test_loss = None):

        self.train_obj = [train_obj] if train_obj else []
        self.test_obj = [test_obj] if test_obj else []
        self.train_loss  = [train_loss] if train_loss  else []
        self.test_loss = [test_loss] if test_loss else []

    
    def update(self, stats):
        for attr_name in self.__dict__.keys():
            setattr(self, attr_name, self.__getattribute__(attr_name) + stats.__getattribute__(attr_name))



    def printout(self, filename, time): 
        print('Stats for %s CPU-time %.4f' % (filename, time))
        print('R2 = %.4f (+/- %.4f) Constraint = %.4f (+/- %.4f)' % \
              (np.mean(self.train_obj), np.std(self.train_obj),
               np.mean(self.train_loss), np.std(self.train_loss)))
        print('Q2 = %.4f (+/- %.4f) Constraint = %.4f (+/- %.4f)'% \
              (np.mean(self.test_obj), np.std(self.test_obj),
               np.mean(self.test_loss), np.std(self.test_loss))
              )
    
    def printout_train(self):
        print("train = %.2f test = %.2f loss-train = %6f loss-test = %.6f" % \
              (self.train_obj[0],
               self.test_obj[0],
               self.train_loss[0],
               self.test_loss[0]))
