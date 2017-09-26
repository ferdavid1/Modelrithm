
import threading 
import time

exitFlag = 0

class CThread (threading.Thread):

	def __init__(self, threadID, name, counter):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.counter = counter

	def run(self):

		print("Starting {} Classifier in a new thread...".format(self.name))
		print_time(self.name, self.counter, 5)
		print("Exiting thread for Classifier: {}".format(self.name))

	def print_time(threadName, counter, delay):
		while counter:
			if exitFlag:
				threadName.exit()
			time.sleep(delay)
			print('{}: {}'.format(threadName, time.ctime(time.time())))
		counter -= 1
	
