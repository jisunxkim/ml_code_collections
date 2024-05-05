import signal
import time

def my_interrupt_handler(signum, frame):
    print('you are tring to interrupt me by pressing ctr-c !!!\n')

signal.signal(signal.SIGINT, my_interrupt_handler)
i = 0
while True and i < 20:
    print("hey...")
    time.sleep(1)
    i +=1

print("completed job without terminated")
